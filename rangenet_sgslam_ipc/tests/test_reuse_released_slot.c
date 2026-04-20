/*
 * Regression: if the consumer releases an older slot after the producer has
 * walked past it, the next publish should reuse that free slot instead of
 * overwriting a newer unread frame.
 *
 * Scenario with slot_count = 4:
 *   - Publish frame 0; consumer acquires it from slot 0 and holds it.
 *   - Publish frames 1, 2, 3, 4 while slot 0 is held.
 *     This leaves unread frames 2, 3, 4 resident; frame 1 was already dropped.
 *   - Consumer releases slot 0, making it free again while the producer cursor
 *     points at slot 2.
 *   - The next publish must reuse slot 0 for frame 5. If it instead follows
 *     the old cursor-only policy, it would overwrite frame 2 and the consumer
 *     would incorrectly jump to frame 3.
 */

#define _POSIX_C_SOURCE 200809L

#include "rnsg_ipc.h"

#include <stdio.h>
#include <stdlib.h>

#define RING_NAME "/rnsg_test_reuse_released_slot"
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
    v.points[0] = (float)fid;
    v.labels[0] = (int32_t)fid;
    check("publish",
          rnsg_producer_publish(r, /*num_points=*/1,
                                /*capture_ns=*/fid, fid, 0) == RNSG_OK);
}

int main(void) {
    rnsg_unlink(RING_NAME);

    rnsg_ring *prod = NULL;
    rnsg_ring *cons = NULL;
    check("create", rnsg_create(RING_NAME, SLOT_COUNT, CAPACITY, &prod) == RNSG_OK);
    check("open", rnsg_open(RING_NAME, &cons) == RNSG_OK);

    publish_frame(prod, 0);

    rnsg_frame_view fv;
    check("acquire0", rnsg_consumer_acquire(cons, 1000000000ll, &fv) == RNSG_OK);
    check("frame0", fv.frame_id == 0);
    check("skip0", fv.skipped_before == 0);
    check("consumed0", fv.consumed_index == 0);

    /* Keep slot 0 held while the producer runs ahead and laps the ring. */
    publish_frame(prod, 1);
    publish_frame(prod, 2);
    publish_frame(prod, 3);
    publish_frame(prod, 4);

    check("release0", rnsg_consumer_release(cons) == RNSG_OK);

    /* This must reuse the newly-freed slot 0 rather than overwriting frame 2. */
    publish_frame(prod, 5);

    /* Frame 1 was already lost while slot 0 was held, so frame 2 is now the
     * oldest survivor. If the producer wrongly overwrote it, this would be 3. */
    check("acquire2", rnsg_consumer_acquire(cons, 0, &fv) == RNSG_OK);
    check("frame2", fv.frame_id == 2);
    check("skip_to_2", fv.skipped_before == 1);
    check("consumed1", fv.consumed_index == 1);
    check("release2", rnsg_consumer_release(cons) == RNSG_OK);

    check("acquire3", rnsg_consumer_acquire(cons, 0, &fv) == RNSG_OK);
    check("frame3", fv.frame_id == 3);
    check("skip3", fv.skipped_before == 0);
    check("consumed2", fv.consumed_index == 2);
    check("release3", rnsg_consumer_release(cons) == RNSG_OK);

    check("acquire4", rnsg_consumer_acquire(cons, 0, &fv) == RNSG_OK);
    check("frame4", fv.frame_id == 4);
    check("skip4", fv.skipped_before == 0);
    check("consumed3", fv.consumed_index == 3);
    check("release4", rnsg_consumer_release(cons) == RNSG_OK);

    check("acquire5", rnsg_consumer_acquire(cons, 0, &fv) == RNSG_OK);
    check("frame5", fv.frame_id == 5);
    check("skip5", fv.skipped_before == 0);
    check("consumed4", fv.consumed_index == 4);
    check("release5", rnsg_consumer_release(cons) == RNSG_OK);

    check("drained", rnsg_consumer_acquire(cons, 0, &fv) == RNSG_TIMEOUT);
    check("drop_count", rnsg_drop_count(prod) == 1);

    rnsg_close(cons);
    rnsg_close(prod);
    check("unlink", rnsg_unlink(RING_NAME) == RNSG_OK);
    printf("test_reuse_released_slot OK\n");
    return 0;
}
