/*
 * SPSC drop-oldest shared-memory ring with zero-copy acquire/release.
 * See include/rnsg_ipc.h for the public contract.
 *
 * Layout of the single POSIX shm region named by the caller:
 *
 *   [ struct rnsg_ctl ]               fixed-size control block
 *   [ slot_seq[slot_count] ]           flexible atomic array (per-slot state)
 *   [ pad to page ]
 *   [ slot 0: header + points + labels ]
 *   [ slot 1: ... ]
 *   ...
 *
 * Per-slot atomic encoding (slot_seq[i]):
 *   - 0                          : empty / mid-write / freshly released
 *   - (seq + 1)                  : ready, available for consumer acquire
 *   - HELD_BIT | (seq + 1)       : currently held by consumer; producer must skip
 *
 * The +1 offset reserves 0 as the "no readable frame" sentinel.
 *
 * Producer publish:
 *   - Walk a private cursor, skipping any slot whose HELD_BIT is set.
 *   - CAS the chosen slot's atom to 0 to claim it for writing (this also
 *     hides any old seq from the consumer's scan).
 *   - Write the header and payload.
 *   - Release-store (head_seq + 1) into the slot's atom to publish.
 *   - Release-store new head_seq.
 *   - sem_post.
 *
 * Consumer acquire:
 *   - Wait on sem until head_seq > consumer_next.
 *   - Scan all slots for the lowest seq >= consumer_next that isn't HELD or 0.
 *   - CAS that slot's atom to set HELD_BIT. Failure means producer just
 *     overwrote it; retry the search.
 *   - Read header from shm; return view with const pointers into shm.
 *
 * Consumer release:
 *   - Release-store 0 into the held slot's atom; producer may now overwrite.
 *
 * Held-slot invariant:
 *   - Once the consumer's CAS succeeds, the slot's HELD_BIT is set.
 *     Producer's claim CAS will fail (or its skip-test will skip) for as
 *     long as HELD_BIT remains set, so the slot's bytes are stable for the
 *     duration of the hold.
 *
 * Drop-oldest accounting:
 *   - Detected by the consumer at acquire time as
 *     skipped = best_seq - consumer_next, when best_seq advances faster than
 *     consumer_next does. Added to the global drop_count atomic.
 */

#define _GNU_SOURCE

#include "rnsg_ipc.h"

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define RNSG_PAGE_ALIGN  4096u
#define RNSG_SLOT_ALIGN  64u
#define RNSG_HELD_BIT    (1ull << 63)
#define RNSG_SEQ_MASK    (~RNSG_HELD_BIT)

struct rnsg_ctl {
    uint32_t magic;
    uint32_t version;
    uint32_t slot_count;
    uint32_t reserved0;
    uint64_t slot_bytes;
    uint64_t capacity_points;
    uint64_t data_offset;
    uint64_t total_bytes;

    sem_t    sem;                  /* posted by producer on every publish */

    _Atomic uint64_t head_seq;     /* monotonic publish counter (producer-only writer) */
    _Atomic uint64_t tail_seq;     /* observability mirror of consumer_next (consumer-only writer) */
    _Atomic uint64_t drop_count;   /* total frames overwritten before consumer could read them */

    /* slot_count entries; see encoding above */
    _Atomic uint64_t slot_seq[];
};

struct rnsg_ring {
    char     name[256];
    int      owner;
    void    *base;
    size_t   map_bytes;
    struct rnsg_ctl *ctl;
    uint8_t *data_arena;

    /* producer-side state */
    uint32_t producer_cursor;      /* next slot index to attempt for publishing */
    int      producer_leased;
    uint32_t producer_leased_idx;  /* slot currently being filled */

    /* consumer-side state */
    uint64_t consumer_next;        /* next producer sequence number to acquire */
    uint64_t acquired_count;       /* dense per-consumer count of successful acquires */
    uint32_t held_slot_idx;        /* RNSG_NO_HELD_SLOT if nothing held */
    uint64_t held_seq;
};

/* ----------------------------------------------------------------- helpers */

static inline size_t align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

static inline int is_pow2_u32(uint32_t x) {
    return x && ((x & (x - 1)) == 0);
}

static int validate_shm_name(const char *name) {
    if (!name || name[0] != '/') return -1;
    size_t n = strlen(name);
    if (n < 2 || n >= 240) return -1;
    for (const char *p = name + 1; *p; ++p) {
        if (*p == '/') return -1;
    }
    return 0;
}

static void compute_layout(uint32_t slot_count,
                           uint32_t capacity_points,
                           size_t *out_slot_bytes,
                           size_t *out_data_offset,
                           size_t *out_total_bytes) {
    size_t hdr = sizeof(rnsg_slot_header);
    size_t pts = (size_t)capacity_points * 4u * sizeof(float);
    size_t lbl = (size_t)capacity_points * sizeof(int32_t);
    size_t slot = align_up(hdr + pts + lbl, RNSG_SLOT_ALIGN);

    size_t ctl_fixed = sizeof(struct rnsg_ctl)
                     + (size_t)slot_count * sizeof(_Atomic uint64_t);
    size_t data_off = align_up(ctl_fixed, RNSG_PAGE_ALIGN);
    size_t total = align_up(data_off + slot * (size_t)slot_count, RNSG_PAGE_ALIGN);

    *out_slot_bytes = slot;
    *out_data_offset = data_off;
    *out_total_bytes = total;
}

static inline uint8_t *slot_ptr(struct rnsg_ring *r, uint32_t idx) {
    return r->data_arena + (size_t)idx * (size_t)r->ctl->slot_bytes;
}

static uint64_t now_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ----------------------------------------------------------------- lifecycle */

rnsg_status rnsg_create(const char *name,
                        uint32_t slot_count,
                        uint32_t capacity_points,
                        rnsg_ring **out) {
    if (!out || validate_shm_name(name) != 0) return RNSG_E_INVAL;

    if (slot_count == 0)      slot_count = RNSG_DEFAULT_SLOT_COUNT;
    if (capacity_points == 0) capacity_points = RNSG_DEFAULT_CAPACITY_POINTS;
    if (!is_pow2_u32(slot_count) || slot_count < 2) return RNSG_E_INVAL;
    if (capacity_points > (1u << 24)) return RNSG_E_INVAL;

    size_t slot_bytes, data_off, total_bytes;
    compute_layout(slot_count, capacity_points, &slot_bytes, &data_off, &total_bytes);

    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        if (errno == EEXIST) return RNSG_E_EXIST;
        return RNSG_E_OS;
    }
    if (ftruncate(fd, (off_t)total_bytes) != 0) {
        int saved = errno;
        close(fd);
        shm_unlink(name);
        errno = saved;
        return RNSG_E_OS;
    }

    void *base = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        shm_unlink(name);
        return RNSG_E_OS;
    }

    memset(base, 0, total_bytes);
    struct rnsg_ctl *ctl = (struct rnsg_ctl *)base;
    ctl->magic           = RNSG_MAGIC;
    ctl->version         = RNSG_VERSION;
    ctl->slot_count      = slot_count;
    ctl->slot_bytes      = (uint64_t)slot_bytes;
    ctl->capacity_points = (uint64_t)capacity_points;
    ctl->data_offset     = (uint64_t)data_off;
    ctl->total_bytes     = (uint64_t)total_bytes;

    if (sem_init(&ctl->sem, /*pshared=*/1, /*value=*/0) != 0) {
        munmap(base, total_bytes);
        shm_unlink(name);
        return RNSG_E_OS;
    }

    atomic_store_explicit(&ctl->head_seq,   0, memory_order_relaxed);
    atomic_store_explicit(&ctl->tail_seq,   0, memory_order_relaxed);
    atomic_store_explicit(&ctl->drop_count, 0, memory_order_relaxed);
    for (uint32_t i = 0; i < slot_count; ++i) {
        atomic_store_explicit(&ctl->slot_seq[i], 0, memory_order_relaxed);
    }

    rnsg_ring *r = (rnsg_ring *)calloc(1, sizeof(*r));
    if (!r) {
        sem_destroy(&ctl->sem);
        munmap(base, total_bytes);
        shm_unlink(name);
        return RNSG_E_NOMEM;
    }
    snprintf(r->name, sizeof r->name, "%s", name);
    r->owner          = 1;
    r->base           = base;
    r->map_bytes      = total_bytes;
    r->ctl            = ctl;
    r->data_arena     = (uint8_t *)base + data_off;
    r->producer_cursor = 0;
    r->held_slot_idx  = RNSG_NO_HELD_SLOT;

    *out = r;
    return RNSG_OK;
}

rnsg_status rnsg_open(const char *name, rnsg_ring **out) {
    if (!out || validate_shm_name(name) != 0) return RNSG_E_INVAL;

    int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0) {
        if (errno == ENOENT) return RNSG_E_NOENT;
        return RNSG_E_OS;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(struct rnsg_ctl)) {
        close(fd);
        return RNSG_E_VERSION;
    }
    size_t map_bytes = (size_t)st.st_size;

    void *base = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return RNSG_E_OS;

    struct rnsg_ctl *ctl = (struct rnsg_ctl *)base;
    if (ctl->magic != RNSG_MAGIC || ctl->version != RNSG_VERSION
        || ctl->total_bytes != map_bytes) {
        munmap(base, map_bytes);
        return RNSG_E_VERSION;
    }

    rnsg_ring *r = (rnsg_ring *)calloc(1, sizeof(*r));
    if (!r) {
        munmap(base, map_bytes);
        return RNSG_E_NOMEM;
    }
    snprintf(r->name, sizeof r->name, "%s", name);
    r->owner          = 0;
    r->base           = base;
    r->map_bytes      = map_bytes;
    r->ctl            = ctl;
    r->data_arena     = (uint8_t *)base + ctl->data_offset;
    r->producer_cursor = 0;
    r->held_slot_idx  = RNSG_NO_HELD_SLOT;
    r->consumer_next  = atomic_load_explicit(&ctl->head_seq, memory_order_acquire);
    /* New consumers start from the live edge; they don't replay backlog. */

    *out = r;
    return RNSG_OK;
}

void rnsg_close(rnsg_ring *r) {
    if (!r) return;
    /* Release any held slot so we don't permanently lose ring capacity. */
    if (r->held_slot_idx != RNSG_NO_HELD_SLOT && r->ctl) {
        atomic_store_explicit(&r->ctl->slot_seq[r->held_slot_idx],
                              0, memory_order_release);
        r->held_slot_idx = RNSG_NO_HELD_SLOT;
    }
    if (r->base) munmap(r->base, r->map_bytes);
    free(r);
}

rnsg_status rnsg_unlink(const char *name) {
    if (validate_shm_name(name) != 0) return RNSG_E_INVAL;
    if (shm_unlink(name) == 0) return RNSG_OK;
    if (errno == ENOENT) return RNSG_E_NOENT;
    return RNSG_E_OS;
}

/* ----------------------------------------------------------------- introspection */

uint32_t rnsg_slot_count(const rnsg_ring *r)      { return r ? r->ctl->slot_count : 0; }
uint32_t rnsg_capacity_points(const rnsg_ring *r) { return r ? (uint32_t)r->ctl->capacity_points : 0; }
uint64_t rnsg_slot_bytes(const rnsg_ring *r)      { return r ? r->ctl->slot_bytes : 0; }

uint64_t rnsg_drop_count(const rnsg_ring *r) {
    return r ? atomic_load_explicit(&r->ctl->drop_count, memory_order_relaxed) : 0;
}
uint64_t rnsg_head_seq(const rnsg_ring *r) {
    return r ? atomic_load_explicit(&r->ctl->head_seq, memory_order_acquire) : 0;
}
uint64_t rnsg_tail_seq(const rnsg_ring *r) {
    return r ? atomic_load_explicit(&r->ctl->tail_seq, memory_order_acquire) : 0;
}

/* ----------------------------------------------------------------- producer */

rnsg_status rnsg_producer_lease(rnsg_ring *r, rnsg_slot_view *out) {
    if (!r || !out) return RNSG_E_INVAL;
    if (r->producer_leased) return RNSG_E_BUSY;

    uint32_t slot_count = r->ctl->slot_count;

    /* Walk the cursor forward, skipping any slot whose HELD_BIT is set.
     * Bound the walk to slot_count attempts; with SPSC and at most one held
     * slot we always succeed in <= 2 attempts, but we also defend against
     * a buggy consumer that holds more than expected. */
    for (uint32_t tries = 0; tries < slot_count; ++tries) {
        uint32_t slot_idx = r->producer_cursor;
        r->producer_cursor = (r->producer_cursor + 1u) & (slot_count - 1u);

        uint64_t cur = atomic_load_explicit(&r->ctl->slot_seq[slot_idx],
                                            memory_order_acquire);
        if (cur & RNSG_HELD_BIT) continue;

        /* Try to claim by setting the slot to "writing" (0). The CAS races
         * only with the consumer's acquire CAS on this same slot. On
         * failure, loop and consider the next cursor position. */
        uint64_t expected = cur;
        if (atomic_compare_exchange_strong_explicit(
                &r->ctl->slot_seq[slot_idx],
                &expected, 0,
                memory_order_acq_rel, memory_order_relaxed)) {
            uint8_t *p = slot_ptr(r, slot_idx);
            rnsg_slot_header *hdr = (rnsg_slot_header *)p;
            hdr->magic         = RNSG_MAGIC;
            hdr->version       = RNSG_VERSION;
            hdr->points_offset = (uint64_t)sizeof(rnsg_slot_header);
            hdr->labels_offset = hdr->points_offset
                               + (uint64_t)r->ctl->capacity_points * 4u * sizeof(float);

            out->slot_idx        = slot_idx;
            out->capacity_points = (uint32_t)r->ctl->capacity_points;
            out->header          = hdr;
            out->points          = (float *)(p + hdr->points_offset);
            out->labels          = (int32_t *)(p + hdr->labels_offset);

            r->producer_leased     = 1;
            r->producer_leased_idx = slot_idx;
            return RNSG_OK;
        }
        /* CAS failed: the value was changed (consumer acquired). Try next slot. */
    }

    return RNSG_E_FULL;
}

rnsg_status rnsg_producer_publish(rnsg_ring *r,
                                  uint32_t num_points,
                                  uint64_t capture_ns,
                                  uint64_t frame_id,
                                  uint32_t flags) {
    if (!r) return RNSG_E_INVAL;
    if (!r->producer_leased) return RNSG_E_BUSY;
    if (num_points > r->ctl->capacity_points) return RNSG_E_TOO_LARGE;

    uint32_t slot_idx = r->producer_leased_idx;
    uint64_t my_seq   = atomic_load_explicit(&r->ctl->head_seq,
                                             memory_order_relaxed);

    uint8_t *p = slot_ptr(r, slot_idx);
    rnsg_slot_header *hdr = (rnsg_slot_header *)p;
    hdr->frame_id   = (frame_id == RNSG_INVALID_FRAME_ID) ? my_seq : frame_id;
    hdr->capture_ns = capture_ns;
    hdr->publish_ns = now_monotonic_ns();
    hdr->num_points = num_points;
    hdr->flags      = flags;

    /* Publish: mark slot ready, advance head, wake any waiter. */
    atomic_store_explicit(&r->ctl->slot_seq[slot_idx],
                          my_seq + 1u, memory_order_release);
    atomic_store_explicit(&r->ctl->head_seq,
                          my_seq + 1u, memory_order_release);
    sem_post(&r->ctl->sem);

    r->producer_leased = 0;
    return RNSG_OK;
}

/* ----------------------------------------------------------------- consumer */

static int sem_wait_with_timeout(sem_t *sem, int64_t timeout_ns) {
    if (timeout_ns < 0) {
        while (sem_wait(sem) != 0) {
            if (errno != EINTR) return -1;
        }
        return 0;
    }
    if (timeout_ns == 0) {
        if (sem_trywait(sem) == 0) return 0;
        return -1;  /* errno = EAGAIN if empty */
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += (time_t)(timeout_ns / 1000000000ll);
    ts.tv_nsec += (long)(timeout_ns % 1000000000ll);
    if (ts.tv_nsec >= 1000000000l) {
        ts.tv_sec  += 1;
        ts.tv_nsec -= 1000000000l;
    }
    while (sem_timedwait(sem, &ts) != 0) {
        if (errno == EINTR) continue;
        return -1;  /* errno = ETIMEDOUT or other */
    }
    return 0;
}

rnsg_status rnsg_consumer_acquire(rnsg_ring *r,
                                  int64_t timeout_ns,
                                  rnsg_frame_view *out_view) {
    if (!r || !out_view) return RNSG_E_INVAL;
    if (r->held_slot_idx != RNSG_NO_HELD_SLOT) return RNSG_E_BUSY;

    uint32_t slot_count = r->ctl->slot_count;

    for (;;) {
        uint64_t head = atomic_load_explicit(&r->ctl->head_seq,
                                             memory_order_acquire);
        if (head <= r->consumer_next) {
            if (sem_wait_with_timeout(&r->ctl->sem, timeout_ns) != 0) {
                if (errno == ETIMEDOUT || errno == EAGAIN) return RNSG_TIMEOUT;
                return RNSG_E_OS;
            }
            continue;
        }

        /* Find the lowest seq >= consumer_next currently resident in the ring. */
        uint64_t best_seq  = UINT64_MAX;
        uint32_t best_slot = RNSG_NO_HELD_SLOT;
        for (uint32_t i = 0; i < slot_count; ++i) {
            uint64_t v = atomic_load_explicit(&r->ctl->slot_seq[i],
                                              memory_order_acquire);
            if (v == 0) continue;
            if (v & RNSG_HELD_BIT) continue;  /* shouldn't happen with single consumer */
            uint64_t seq = v - 1u;
            if (seq >= r->consumer_next && seq < best_seq) {
                best_seq  = seq;
                best_slot = i;
            }
        }

        if (best_slot == RNSG_NO_HELD_SLOT) {
            /* All eligible slots are mid-write. Brief spin and re-scan.
             * In practice this resolves within a single producer publish. */
            sched_yield();
            continue;
        }

        /* CAS to claim. Failure means the producer overwrote this slot
         * between our scan and our CAS. Retry the search. */
        uint64_t expected = best_seq + 1u;
        uint64_t desired  = expected | RNSG_HELD_BIT;
        if (!atomic_compare_exchange_strong_explicit(
                &r->ctl->slot_seq[best_slot],
                &expected, desired,
                memory_order_acquire, memory_order_relaxed)) {
            sched_yield();
            continue;
        }

        /* We now own the slot. Read header from shm. */
        uint8_t *p = slot_ptr(r, best_slot);
        rnsg_slot_header *hdr = (rnsg_slot_header *)p;

        uint64_t skipped = best_seq - r->consumer_next;
        if (skipped > 0) {
            atomic_fetch_add_explicit(&r->ctl->drop_count, skipped,
                                      memory_order_relaxed);
        }

        out_view->slot_idx       = best_slot;
        out_view->num_points     = hdr->num_points;
        out_view->frame_id       = hdr->frame_id;
        out_view->consumed_index = r->acquired_count;
        out_view->capture_ns     = hdr->capture_ns;
        out_view->publish_ns     = hdr->publish_ns;
        out_view->flags          = hdr->flags;
        out_view->reserved0      = 0;
        out_view->skipped_before = skipped;
        out_view->points         = (const float *)(p + hdr->points_offset);
        out_view->labels         = (const int32_t *)(p + hdr->labels_offset);

        r->held_slot_idx = best_slot;
        r->held_seq      = best_seq;
        r->consumer_next = best_seq + 1u;
        r->acquired_count += 1u;
        atomic_store_explicit(&r->ctl->tail_seq,
                              r->consumer_next, memory_order_release);
        return RNSG_OK;
    }
}

rnsg_status rnsg_consumer_release(rnsg_ring *r) {
    if (!r) return RNSG_E_INVAL;
    if (r->held_slot_idx == RNSG_NO_HELD_SLOT) return RNSG_E_BUSY;

    /* Release-store 0 so any earlier reads from the slot happen-before the
     * producer's subsequent claim+write. */
    atomic_store_explicit(&r->ctl->slot_seq[r->held_slot_idx],
                          0, memory_order_release);
    r->held_slot_idx = RNSG_NO_HELD_SLOT;
    return RNSG_OK;
}
