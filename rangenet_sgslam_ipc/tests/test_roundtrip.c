/*
 * Single-process round-trip with the zero-copy acquire/release API.
 *
 * Producer leases a slot, fills it with synthetic data, publishes; consumer
 * acquires, verifies the view aliases shm by checking deterministic content,
 * releases. Repeated for several frames in order.
 */

#define _POSIX_C_SOURCE 200809L

#include "rnsg_ipc.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RING_NAME "/rnsg_test_roundtrip"
#define SLOT_COUNT 16u
#define CAPACITY 1024u

static void fail(const char *what, rnsg_status rc) {
    fprintf(stderr, "FAIL %s rc=%d errno=%d\n", what, rc, errno);
    rnsg_unlink(RING_NAME);
    exit(1);
}

int main(void) {
    rnsg_unlink(RING_NAME);

    rnsg_ring *prod = NULL, *cons = NULL;
    rnsg_status rc = rnsg_create(RING_NAME, SLOT_COUNT, CAPACITY, &prod);
    if (rc != RNSG_OK) fail("create", rc);
    rc = rnsg_open(RING_NAME, &cons);
    if (rc != RNSG_OK) fail("open", rc);

    if (rnsg_slot_count(prod) != SLOT_COUNT) fail("slot_count", RNSG_E_IO);
    if (rnsg_capacity_points(prod) != CAPACITY) fail("capacity", RNSG_E_IO);

    for (uint64_t f = 0; f < 8; ++f) {
        rnsg_slot_view sv;
        rc = rnsg_producer_lease(prod, &sv);
        if (rc != RNSG_OK) fail("lease", rc);

        uint32_t n = (uint32_t)((f + 1) * 64);
        for (uint32_t i = 0; i < n; ++i) {
            sv.points[4*i + 0] = (float)(f * 1000 + i);
            sv.points[4*i + 1] = (float)(f * 1000 + i) + 0.25f;
            sv.points[4*i + 2] = (float)(f * 1000 + i) + 0.50f;
            sv.points[4*i + 3] = (float)(f * 1000 + i) + 0.75f;
            sv.labels[i] = (int32_t)((f << 16) | i);
        }
        rc = rnsg_producer_publish(prod, n, /*capture_ns=*/1000ull + f,
                                   /*frame_id=*/100ull + f,
                                   RNSG_FLAG_RAW_SEMANTICKITTI_LABELS);
        if (rc != RNSG_OK) fail("publish", rc);

        rnsg_frame_view fv;
        rc = rnsg_consumer_acquire(cons, /*timeout_ns=*/1000000000ll, &fv);
        if (rc != RNSG_OK) fail("acquire", rc);
        if (fv.frame_id != 100ull + f) fail("frame_id", RNSG_E_IO);
        if (fv.consumed_index != f) fail("consumed_index", RNSG_E_IO);
        if (fv.capture_ns != 1000ull + f) fail("capture_ns", RNSG_E_IO);
        if (fv.num_points != n) fail("num_points", RNSG_E_IO);
        if ((fv.flags & RNSG_FLAG_RAW_SEMANTICKITTI_LABELS) == 0) fail("flags", RNSG_E_IO);
        if (fv.skipped_before != 0) fail("skipped_before", RNSG_E_IO);
        if (!fv.points || !fv.labels) fail("view_pointers", RNSG_E_IO);

        for (uint32_t i = 0; i < n; ++i) {
            if (fv.points[4*i + 0] != (float)(f * 1000 + i)) fail("pts.x", RNSG_E_IO);
            if (fv.points[4*i + 3] != (float)(f * 1000 + i) + 0.75f) fail("pts.r", RNSG_E_IO);
            if (fv.labels[i] != (int32_t)((f << 16) | i)) fail("label", RNSG_E_IO);
        }

        rc = rnsg_consumer_release(cons);
        if (rc != RNSG_OK) fail("release", rc);
    }

    if (rnsg_drop_count(prod) != 0) fail("drop_count_nonzero", RNSG_E_IO);

    /* Acquire when held should fail BUSY. */
    {
        rnsg_slot_view sv;
        if (rnsg_producer_lease(prod, &sv) != RNSG_OK) fail("lease2", RNSG_E_IO);
        if (rnsg_producer_publish(prod, 1, 0, 999, 0) != RNSG_OK) fail("publish2", RNSG_E_IO);

        rnsg_frame_view fv;
        if (rnsg_consumer_acquire(cons, 0, &fv) != RNSG_OK) fail("acq3", RNSG_E_IO);
        rnsg_frame_view fv2;
        if (rnsg_consumer_acquire(cons, 0, &fv2) != RNSG_E_BUSY) fail("acq_busy", RNSG_E_IO);
        if (rnsg_consumer_release(cons) != RNSG_OK) fail("rel3", RNSG_E_IO);
        if (rnsg_consumer_release(cons) != RNSG_E_BUSY) fail("rel_busy", RNSG_E_IO);
    }

    /* Timeout paths */
    rnsg_frame_view fv;
    if (rnsg_consumer_acquire(cons, /*timeout_ns=*/0, &fv) != RNSG_TIMEOUT)
        fail("nonblocking_empty", RNSG_E_IO);
    if (rnsg_consumer_acquire(cons, /*timeout_ns=*/10000000ll, &fv) != RNSG_TIMEOUT)
        fail("blocking_timeout", RNSG_E_IO);

    rnsg_close(cons);
    rnsg_close(prod);
    if (rnsg_unlink(RING_NAME) != RNSG_OK) fail("unlink", RNSG_E_IO);

    printf("test_roundtrip OK\n");
    return 0;
}
