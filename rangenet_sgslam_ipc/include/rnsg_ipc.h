/*
 * rnsg_ipc.h - shared-memory FIFO contract between RangeNet and SG-SLAM.
 *
 * Wire format and entry points are stable across the Python producer (today,
 * via ctypes) and the C++ consumer (today, via direct linkage).
 *
 * Variant: SPSC, drop-oldest, zero-copy with acquire/release lifetime.
 *   - Producer never blocks. It first reuses any free slot; only when every
 *     unheld slot still contains unread data does it overwrite the oldest
 *     non-held slot.
 *   - Consumer holds at most one slot at a time. While held, the producer
 *     is forbidden from overwriting that slot.
 *   - Consumer reads the slot's payload directly from shared memory through
 *     const pointers; no bytes are copied by the IPC layer.
 *
 * Memory ordering and atomic state are kept entirely inside librnsg_ipc.so;
 * no _Atomic types appear in this header so it is safe to include from C++.
 *
 * Per-process model:
 *   - The owner (typically the producer) calls rnsg_create() once to allocate
 *     the named shm region.
 *   - Any peer (typically the consumer) calls rnsg_open() to attach.
 *   - Either side may call rnsg_unlink() at shutdown to remove the shm name.
 *
 * Multi-producer or multi-consumer use is unsupported by design (SPSC).
 */

#ifndef RNSG_IPC_H
#define RNSG_IPC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- versioning */

#define RNSG_MAGIC   0x47534E52u  /* 'RNSG' little-endian */
#define RNSG_VERSION 2u           /* bumped: acquire/release zero-copy */

/* Sentinel for "library, please assign a frame_id from the publish counter" */
#define RNSG_INVALID_FRAME_ID UINT64_MAX

/* Sentinel for "no slot currently held" */
#define RNSG_NO_HELD_SLOT UINT32_MAX

/*
 * Defaults; callers may override at create time.
 *
 * RNSG_DEFAULT_SLOT_COUNT is sized to absorb p99 SG-SLAM front-end latency
 * (~400 ms on the AWS dev box) at the 10 Hz scan rate without dropping
 * frames the consumer hasn't yet seen. See Docs/Design/contract.md for the
 * sizing derivation and measured latency distribution.
 */
#define RNSG_DEFAULT_SLOT_COUNT      8u
#define RNSG_DEFAULT_CAPACITY_POINTS 200000u

/* Flags for rnsg_producer_publish() */
#define RNSG_FLAG_RAW_SEMANTICKITTI_LABELS 0x1u

/* ---------------------------------------------------------------- error codes */

typedef enum {
    RNSG_OK             =  0,
    RNSG_TIMEOUT        =  1,  /* consumer acquire exceeded its deadline */
    RNSG_E_INVAL        = -1,
    RNSG_E_NOENT        = -2,
    RNSG_E_EXIST        = -3,
    RNSG_E_NOMEM        = -4,
    RNSG_E_IO           = -5,
    RNSG_E_TOO_LARGE    = -6,  /* num_points > slot capacity */
    RNSG_E_VERSION      = -7,  /* shm region magic/version mismatch */
    RNSG_E_OS           = -8,  /* generic syscall failure; see errno */
    RNSG_E_BUSY         = -9,  /* acquire while already holding, or release while not */
    RNSG_E_FULL         = -10  /* every slot is held (cannot occur in SPSC) */
} rnsg_status;

/* ---------------------------------------------------------------- opaque types */

typedef struct rnsg_ring rnsg_ring;

/*
 * Per-slot header carried in the data region. Exposed here so the consumer
 * may inspect timestamps and frame ids without going through the library.
 * Layout is fixed; do not rearrange. All multi-byte fields are little-endian.
 */
typedef struct rnsg_slot_header {
    uint32_t magic;          /* RNSG_MAGIC */
    uint32_t version;        /* RNSG_VERSION */
    uint64_t frame_id;       /* monotonic, set by producer */
    uint64_t capture_ns;     /* sensor capture timestamp (CLOCK_REALTIME or arbitrary domain) */
    uint64_t publish_ns;     /* CLOCK_MONOTONIC ns at publish */
    uint32_t num_points;     /* <= capacity_points */
    uint32_t flags;          /* RNSG_FLAG_* */
    uint64_t points_offset;  /* bytes from header start to float[num_points][4] (xyz + remission) */
    uint64_t labels_offset;  /* bytes from header start to int32_t[num_points] raw label per point */
    uint8_t  reserved[16];
} rnsg_slot_header;

/*
 * Producer-side view over the slot the producer is currently writing.
 * Pointers remain valid until the matching rnsg_producer_publish() (or the
 * next rnsg_producer_lease() call, or rnsg_close()).
 */
typedef struct rnsg_slot_view {
    uint32_t          slot_idx;
    uint32_t          capacity_points;
    rnsg_slot_header *header;
    float            *points;   /* capacity_points * 4 floats: x, y, z, remission */
    int32_t          *labels;   /* capacity_points int32 labels */
} rnsg_slot_view;

/*
 * Consumer-side zero-copy view of a held frame.
 *
 * Pointers point directly into shared memory. They are valid only between
 * rnsg_consumer_acquire() and the matching rnsg_consumer_release().
 *
 * The consumer must NOT write through these pointers, and must finish all
 * reads before calling release. After release, the producer may immediately
 * overwrite the slot and the pointers become dangling.
 *
 * Index semantics (important for SG-SLAM-style structural use):
 *   - frame_id        : producer-assigned traceability id. SPARSE from the
 *                       consumer's point of view: it skips by
 *                       (skipped_before + 1) on each successful acquire
 *                       under drop-oldest. Use only as a back-reference to
 *                       the producer's source data.
 *   - consumed_index  : dense per-consumer counter of successful acquires.
 *                       Starts at 0 and increments by exactly 1 every
 *                       successful acquire, even when frames were dropped.
 *                       Use this for structural indexing (pose vectors,
 *                       graph node keys, per-frame artifact filenames).
 *   - skipped_before  : how many producer frames were overwritten between
 *                       the previously consumed frame and this one. Useful
 *                       for adapting motion priors and gating thresholds.
 */
typedef struct rnsg_frame_view {
    uint32_t       slot_idx;
    uint32_t       num_points;
    uint64_t       frame_id;        /* producer traceability id (SPARSE) */
    uint64_t       consumed_index;  /* dense per-consumer counter (0, 1, 2, ...) */
    uint64_t       capture_ns;
    uint64_t       publish_ns;
    uint32_t       flags;
    uint32_t       reserved0;
    uint64_t       skipped_before;
    const float   *points;          /* num_points * 4 floats */
    const int32_t *labels;          /* num_points int32 labels */
} rnsg_frame_view;

/* ---------------------------------------------------------------- lifecycle */

/*
 * Create and initialize a new ring under `name`. `name` must follow POSIX
 * shm rules (start with '/', no other slashes).
 * Fails with RNSG_E_EXIST if the ring already exists; call rnsg_unlink()
 * first if you want to reset.
 *
 * `slot_count` must be a power of two and >= 2; pass 0 to use
 * RNSG_DEFAULT_SLOT_COUNT.
 * `capacity_points` is the maximum points-per-frame this ring will accept;
 * pass 0 to use RNSG_DEFAULT_CAPACITY_POINTS.
 *
 * Note: with N slots and one held by the consumer, the producer's effective
 * working depth is N-1. Pick slot_count >= 2.
 */
rnsg_status rnsg_create(const char *name,
                        uint32_t slot_count,
                        uint32_t capacity_points,
                        rnsg_ring **out);

/* Attach to an existing ring under `name`. */
rnsg_status rnsg_open(const char *name, rnsg_ring **out);

/*
 * Detach. If the local consumer was still holding a slot, this releases it
 * so the producer does not permanently lose a slot of capacity.
 */
void rnsg_close(rnsg_ring *r);

/* Remove the named shm region. Safe to call after the last close. */
rnsg_status rnsg_unlink(const char *name);

/* ---------------------------------------------------------------- introspection */

uint32_t rnsg_slot_count(const rnsg_ring *r);
uint32_t rnsg_capacity_points(const rnsg_ring *r);
uint64_t rnsg_slot_bytes(const rnsg_ring *r);

/*
 * Total number of frames that the producer overwrote before any consumer
 * could acquire them. Reset to 0 only at create-time.
 */
uint64_t rnsg_drop_count(const rnsg_ring *r);

uint64_t rnsg_head_seq(const rnsg_ring *r);  /* total publishes */
uint64_t rnsg_tail_seq(const rnsg_ring *r);  /* total successful acquires (consumer-private mirror) */

/* ---------------------------------------------------------------- producer */

/*
 * Lease the next slot for writing. The same slot is returned for every call
 * until rnsg_producer_publish() (or rnsg_close()) is invoked.
 *
 * The library prefers a free slot (slot_seq == 0), searching in producer
 * cursor order for locality. If none are free, it overwrites the oldest
 * non-held published slot. Held slots are never reused. While leased, the
 * slot remains unreadable to the consumer.
 *
 * Returns RNSG_E_FULL only in the impossible (with SPSC) case where every
 * slot is simultaneously held; caller should treat this as a programmer
 * error in their consumer side.
 */
rnsg_status rnsg_producer_lease(rnsg_ring *r, rnsg_slot_view *out);

/*
 * Commit the currently-leased slot. `num_points` must be <= capacity_points.
 * `frame_id` and `capture_ns` are stamped into the header; pass
 * RNSG_INVALID_FRAME_ID for `frame_id` to let the library assign a monotonic
 * counter starting at 0.
 */
rnsg_status rnsg_producer_publish(rnsg_ring *r,
                                  uint32_t num_points,
                                  uint64_t capture_ns,
                                  uint64_t frame_id,
                                  uint32_t flags);

/* ---------------------------------------------------------------- consumer */

/*
 * Block until the next ready frame can be acquired, then return a zero-copy
 * view into the slot. The consumer must call rnsg_consumer_release() on the
 * same ring handle before calling acquire again.
 *
 *   timeout_ns < 0  -> block forever
 *   timeout_ns == 0 -> non-blocking; returns RNSG_TIMEOUT if nothing ready
 *   timeout_ns > 0  -> bounded wait
 *
 * Returns RNSG_E_BUSY if a slot is already held.
 *
 * out_view->skipped_before reports how many frames were overwritten between
 * the previously released frame and this one (drop-oldest accounting).
 */
rnsg_status rnsg_consumer_acquire(rnsg_ring *r,
                                  int64_t timeout_ns,
                                  rnsg_frame_view *out_view);

/*
 * Release the currently-held slot back to the producer. After this returns
 * the const pointers from the prior acquire become invalid and must not be
 * dereferenced.
 *
 * Returns RNSG_E_BUSY if no slot is currently held.
 */
rnsg_status rnsg_consumer_release(rnsg_ring *r);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* RNSG_IPC_H */
