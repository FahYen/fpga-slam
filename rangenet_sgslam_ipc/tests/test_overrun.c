/*
 * Drop-oldest behavior with no consumer holding any slot:
 * producer publishes 10 frames into a depth-4 ring; the surviving frames
 * are the last 4 (one per slot), and the consumer's drop_count reflects
 * the 6 lost frames at acquire time.
 */

#define _POSIX_C_SOURCE 200809L

#include "rnsg_ipc.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define RING_NAME "/rnsg_test_overrun"
#define SLOT_COUNT 4u
#define CAPACITY   16u

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
    v.labels[0] = (int32_t)fid;
    v.points[0] = (float)fid;
    check("publish",
          rnsg_producer_publish(r, /*num_points=*/1, /*capture_ns=*/fid,
                                fid, 0) == RNSG_OK);
}

static int acquire_one(rnsg_ring *c, rnsg_frame_view *fv) {
    return rnsg_consumer_acquire(c, /*timeout_ns=*/0, fv);
}

int main(void) {
    rnsg_unlink(RING_NAME);
    rnsg_ring *r = NULL;
    check("create", rnsg_create(RING_NAME, SLOT_COUNT, CAPACITY, &r) == RNSG_OK);
    rnsg_ring *c = NULL;
    check("open", rnsg_open(RING_NAME, &c) == RNSG_OK);

    /* Publish 10 frames; producer cycles slots 0..3 and overwrites in place.
     * Surviving content per slot:
     *   slot 0 -> frame 8 (overwritten 0,4,8)
     *   slot 1 -> frame 9 (overwritten 1,5,9)
     *   slot 2 -> frame 6 (overwritten 2,6)
     *   slot 3 -> frame 7 (overwritten 3,7)
     * Lowest survivor seq = 6, so consumer first sees frame 6 and skips 6 frames.
     */
    for (uint64_t f = 0; f < 10; ++f) publish_frame(r, f);

    rnsg_frame_view fv;

    check("first_acquire", acquire_one(c, &fv) == RNSG_OK);
    if (fv.frame_id != 6 || fv.skipped_before != 6 || fv.labels[0] != 6
        || fv.consumed_index != 0) {
        fprintf(stderr, "first frame_id=%llu skipped=%llu label=%d consumed=%llu\n",
                (unsigned long long)fv.frame_id,
                (unsigned long long)fv.skipped_before,
                fv.labels[0],
                (unsigned long long)fv.consumed_index);
        check("first_lap_metadata", 0);
    }
    check("first_release", rnsg_consumer_release(c) == RNSG_OK);

    /* Drain in seq order: 7, 8, 9. consumed_index must be 1, 2, 3 (dense). */
    uint64_t expected_seq[3] = {7, 8, 9};
    uint64_t expected_consumed[3] = {1, 2, 3};
    for (int i = 0; i < 3; ++i) {
        check("acq_drain", acquire_one(c, &fv) == RNSG_OK);
        if (fv.frame_id != expected_seq[i] || fv.skipped_before != 0
            || fv.consumed_index != expected_consumed[i]) {
            fprintf(stderr, "drain[%d]: want frame=%llu got=%llu skipped=%llu consumed=%llu (want %llu)\n",
                    i, (unsigned long long)expected_seq[i],
                    (unsigned long long)fv.frame_id,
                    (unsigned long long)fv.skipped_before,
                    (unsigned long long)fv.consumed_index,
                    (unsigned long long)expected_consumed[i]);
            check("ordered_tail", 0);
        }
        if (fv.labels[0] != (int32_t)expected_seq[i]) check("drain_label", 0);
        check("rel_drain", rnsg_consumer_release(c) == RNSG_OK);
    }

    check("drained_returns_timeout", acquire_one(c, &fv) == RNSG_TIMEOUT);

    if (rnsg_drop_count(r) != 6) {
        fprintf(stderr, "drop_count=%llu want 6\n",
                (unsigned long long)rnsg_drop_count(r));
        check("drop_count", 0);
    }

    rnsg_close(c);
    rnsg_close(r);
    check("unlink", rnsg_unlink(RING_NAME) == RNSG_OK);

    printf("test_overrun OK\n");
    return 0;
}
