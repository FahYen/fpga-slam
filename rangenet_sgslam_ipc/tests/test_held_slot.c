/*
 * Held-slot invariant: while the consumer holds a slot, the producer must
 * never overwrite it, even under heavy publish overrun.
 *
 * Scenario:
 *   - slot_count = 4
 *   - Publish frame 0; consumer acquires it (slot 0 becomes HELD).
 *   - Publish frames 1..10 with no consumer release in between.
 *   - The held view's pointers must still alias frame 0's bytes verbatim.
 *   - Release frame 0; subsequent acquires must observe drop-oldest:
 *     surviving frames in the 3 unheld slots are the latest 3 publishes
 *     (8, 9, 10), and drop_count must equal 7 (frames 1..7 lost).
 *
 * Cursor walk while held (with slot_count=4 and slot 0 held):
 *   frame 1 -> slot 1
 *   frame 2 -> slot 2
 *   frame 3 -> slot 3
 *   frame 4 -> cursor 0 held, skip to slot 1 (overwrites frame 1)
 *   frame 5 -> slot 2 (overwrites frame 2)
 *   frame 6 -> slot 3 (overwrites frame 3)
 *   frame 7 -> cursor 0 held, skip to slot 1 (overwrites frame 4)
 *   frame 8 -> slot 2 (overwrites frame 5)
 *   frame 9 -> slot 3 (overwrites frame 6)
 *   frame 10 -> cursor 0 held, skip to slot 1 (overwrites frame 7)
 * Final: slot 0 = frame 0 (HELD), slot 1 = frame 10, slot 2 = frame 8, slot 3 = frame 9.
 * After release, consumer search finds lowest seq >= 1 in {8, 9, 10} = 8.
 *   skipped_before for that acquire = 8 - 1 = 7.
 */

#define _POSIX_C_SOURCE 200809L

#include "rnsg_ipc.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RING_NAME "/rnsg_test_held_slot"
#define SLOT_COUNT 4u
#define CAPACITY 8u

static void check(const char *what, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", what);
        rnsg_unlink(RING_NAME);
        exit(1);
    }
}

static void publish_frame(rnsg_ring *r, uint64_t fid) {
    rnsg_slot_view v;
    check("lease", rnsg_producer_lease(r, &v) == RNSG_OK);
    /* Fill all CAPACITY points with a deterministic per-frame pattern. */
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        v.points[4*i + 0] = (float)(fid * 1000 + i);
        v.points[4*i + 1] = (float)(fid * 1000 + i) + 0.25f;
        v.points[4*i + 2] = (float)(fid * 1000 + i) + 0.50f;
        v.points[4*i + 3] = (float)(fid * 1000 + i) + 0.75f;
        v.labels[i] = (int32_t)(fid * 100 + i);
    }
    check("publish",
          rnsg_producer_publish(r, /*num_points=*/CAPACITY,
                                /*capture_ns=*/fid, fid, 0) == RNSG_OK);
}

static void verify_frame_content(const rnsg_frame_view *fv, uint64_t fid) {
    check("num_points", fv->num_points == CAPACITY);
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        if (fv->points[4*i + 0] != (float)(fid * 1000 + i)) {
            fprintf(stderr, "frame %llu pt %u x mismatch: got %f want %f\n",
                    (unsigned long long)fid, i,
                    fv->points[4*i + 0], (float)(fid * 1000 + i));
            check("pts.x", 0);
        }
        if (fv->points[4*i + 3] != (float)(fid * 1000 + i) + 0.75f) check("pts.r", 0);
        if (fv->labels[i] != (int32_t)(fid * 100 + i)) check("label", 0);
    }
}

int main(void) {
    rnsg_unlink(RING_NAME);
    rnsg_ring *r = NULL;
    check("create", rnsg_create(RING_NAME, SLOT_COUNT, CAPACITY, &r) == RNSG_OK);
    rnsg_ring *c = NULL;
    check("open", rnsg_open(RING_NAME, &c) == RNSG_OK);

    publish_frame(r, 0);

    rnsg_frame_view held;
    check("acquire0", rnsg_consumer_acquire(c, /*timeout_ns=*/1000000000ll, &held) == RNSG_OK);
    check("frame0_id", held.frame_id == 0);
    check("frame0_skipped", held.skipped_before == 0);
    check("frame0_consumed_index", held.consumed_index == 0);
    verify_frame_content(&held, 0);

    /* Snapshot the held bytes for byte-exact comparison after overrun. */
    float pts_snapshot[CAPACITY * 4];
    int32_t lbl_snapshot[CAPACITY];
    memcpy(pts_snapshot, held.points, sizeof(pts_snapshot));
    memcpy(lbl_snapshot, held.labels, sizeof(lbl_snapshot));

    /* Publish 10 more frames while the consumer holds slot 0. */
    for (uint64_t f = 1; f <= 10; ++f) publish_frame(r, f);

    /* Held view must still alias frame 0's bytes verbatim. */
    if (memcmp(pts_snapshot, held.points, sizeof(pts_snapshot)) != 0) {
        fprintf(stderr, "held points were overwritten during producer overrun\n");
        check("held_points_invariant", 0);
    }
    if (memcmp(lbl_snapshot, held.labels, sizeof(lbl_snapshot)) != 0) {
        fprintf(stderr, "held labels were overwritten during producer overrun\n");
        check("held_labels_invariant", 0);
    }
    verify_frame_content(&held, 0);

    /* Producer publishes during hold count toward total; drop_count is only
     * incremented on consumer-side acquire when a lap is observed. So far the
     * consumer has only acquired frame 0, with skipped_before=0, so drop_count
     * is still 0. */
    check("drop_count_pre_release", rnsg_drop_count(r) == 0);

    check("release0", rnsg_consumer_release(c) == RNSG_OK);

    /* After release, the next acquire must see frame 8 (oldest survivor in
     * slots 1, 2, 3 after the cursor-skip dance), with skipped_before = 7.
     * consumed_index must remain dense: 1 (frame 0 was 0). */
    rnsg_frame_view fv;
    check("acquire_post_release",
          rnsg_consumer_acquire(c, /*timeout_ns=*/0, &fv) == RNSG_OK);
    if (fv.frame_id != 8 || fv.skipped_before != 7 || fv.consumed_index != 1) {
        fprintf(stderr, "post-release: frame_id=%llu skipped=%llu consumed=%llu (want 8, 7, 1)\n",
                (unsigned long long)fv.frame_id,
                (unsigned long long)fv.skipped_before,
                (unsigned long long)fv.consumed_index);
        check("post_release_metadata", 0);
    }
    verify_frame_content(&fv, 8);
    check("rel_post", rnsg_consumer_release(c) == RNSG_OK);

    /* Then frame 9, then frame 10, in seq order; consumed_index 2, 3. */
    check("acq_9", rnsg_consumer_acquire(c, 0, &fv) == RNSG_OK);
    if (fv.frame_id != 9 || fv.skipped_before != 0 || fv.consumed_index != 2)
        check("frame9", 0);
    verify_frame_content(&fv, 9);
    check("rel_9", rnsg_consumer_release(c) == RNSG_OK);

    check("acq_10", rnsg_consumer_acquire(c, 0, &fv) == RNSG_OK);
    if (fv.frame_id != 10 || fv.skipped_before != 0 || fv.consumed_index != 3)
        check("frame10", 0);
    verify_frame_content(&fv, 10);
    check("rel_10", rnsg_consumer_release(c) == RNSG_OK);

    check("drained", rnsg_consumer_acquire(c, 0, &fv) == RNSG_TIMEOUT);

    /* Drop count should be 7 (frames 1..7 were never delivered to the consumer). */
    if (rnsg_drop_count(r) != 7) {
        fprintf(stderr, "drop_count=%llu want 7\n",
                (unsigned long long)rnsg_drop_count(r));
        check("drop_count_total", 0);
    }

    /* Producer can still publish a fresh frame and consumer can still read it.
     * consumed_index keeps incrementing densely: should be 4 here (after 0,1,2,3). */
    publish_frame(r, 100);
    check("acq_after_drain", rnsg_consumer_acquire(c, 0, &fv) == RNSG_OK);
    if (fv.frame_id != 100 || fv.consumed_index != 4) {
        fprintf(stderr, "post-drain: frame_id=%llu consumed=%llu (want 100, 4)\n",
                (unsigned long long)fv.frame_id,
                (unsigned long long)fv.consumed_index);
        check("post_drain_frame_id", 0);
    }
    verify_frame_content(&fv, 100);
    check("rel_after_drain", rnsg_consumer_release(c) == RNSG_OK);

    rnsg_close(c);
    rnsg_close(r);
    check("unlink", rnsg_unlink(RING_NAME) == RNSG_OK);

    printf("test_held_slot OK\n");
    return 0;
}
