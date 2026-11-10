import cv2

from ultralytics import YOLO

from ocr_helper import PlateReader
from tracker_helper import VehicleLogger


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"
VIDEO_PATH = "2.mp4"

OUTPUT_PATH = "detected_video.mp4"
CSV_PATH = "vehicle_log.csv"

DETECTION_CONFIDENCE = 0.15
TRACKER_CONFIG = "bytetrack.yaml"

OCR_RETRY_INTERVAL = 5
OCR_VERIFY_INTERVAL = 5

EVENT_RETENTION_SECONDS = 10.0

# Standard ByteTrack configuration uses a track buffer
# of 30 frames. Allow a little more than that before
# treating a reappearing ID as a new vehicle.
RAW_ID_REUSE_GAP_SECONDS = 1.2

TRIPWIRE_POSITION = 0.65


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def get_side(center_y, line_y):

    if center_y < line_y:
        return "ABOVE"

    return "BELOW"


def create_video_writer(path, fps, width, height):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    return writer


# ============================================================
# INITIALIZATION
# ============================================================

print("[INIT] Loading existing YOLO model...")

model = YOLO(MODEL_PATH)

print("[INIT] Initializing OCR...")

plate_reader = PlateReader()

logger = VehicleLogger(CSV_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise SystemExit(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps <= 0:
    fps = 30.0


frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

if frame_width <= 0 or frame_height <= 0:

    cap.release()

    raise SystemExit(
        "Invalid video dimensions."
    )


writer = create_video_writer(
    OUTPUT_PATH,
    fps,
    frame_width,
    frame_height
)


# ============================================================
# TRACKING STATE
# ============================================================

# ByteTrack ID -> persistent event ID
raw_to_event = {}

# ByteTrack ID -> last frame in which it was detected
raw_id_last_seen = {}

# Event ID -> event state
events = {}

next_event_number = 1

total_frames = 0

in_count = 0
out_count = 0

line_y = int(
    frame_height * TRIPWIRE_POSITION
)

retention_frames = max(
    1,
    int(fps * EVENT_RETENTION_SECONDS)
)

raw_id_reuse_gap_frames = max(
    1,
    int(fps * RAW_ID_REUSE_GAP_SECONDS)
)


# ============================================================
# EVENT CREATION
# ============================================================

def create_event(raw_id, box, frame_number):

    global next_event_number

    event_id = f"EVENT_{next_event_number:06d}"

    next_event_number += 1

    center = get_center(box)

    state = {

        "event_id": event_id,

        "last_box": box,
        "last_center": center,

        "last_seen_frame": frame_number,
        "last_ocr_frame": -OCR_RETRY_INTERVAL,

        "stable_side": get_side(
            center[1],
            line_y
        ),

        "direction": None,
        "counted": False,

        "best_plate": "SCANNING",
        "best_plate_confidence": -1.0
    }

    events[event_id] = state

    raw_to_event[raw_id] = event_id

    raw_id_last_seen[raw_id] = frame_number

    print(
        f"[NEW EVENT] {event_id} | "
        f"ByteTrack ID: {raw_id}"
    )

    return state


# ============================================================
# UPDATE EVENT POSITION
# ============================================================

def update_event_position(state, box, frame_number):

    state["last_box"] = box

    state["last_center"] = get_center(box)

    state["last_seen_frame"] = frame_number


# ============================================================
# OCR UPDATE
# ============================================================

def update_best_plate(state):

    event_id = state["event_id"]

    plate = plate_reader.get_cached_plate(
        event_id
    )

    confidence = plate_reader.get_cached_confidence(
        event_id
    )

    if plate == "SCANNING":
        return

    if confidence > state["best_plate_confidence"]:

        previous_plate = state["best_plate"]

        state["best_plate"] = plate

        state["best_plate_confidence"] = confidence

        print(
            f"[EVENT BEST PLATE] {event_id} | "
            f"{previous_plate} -> {plate} | "
            f"Confidence: {confidence:.4f}"
        )


# ============================================================
# TRIPWIRE CROSSING
# ============================================================

def check_crossing(state, center_y):

    global in_count
    global out_count

    current_side = get_side(
        center_y,
        line_y
    )

    previous_side = state["stable_side"]

    if previous_side is None:

        state["stable_side"] = current_side

        return

    if current_side == previous_side:
        return

    if not state["counted"]:

        if (
            previous_side == "ABOVE"
            and current_side == "BELOW"
        ):

            state["direction"] = "IN"

            in_count += 1

        elif (
            previous_side == "BELOW"
            and current_side == "ABOVE"
        ):

            state["direction"] = "OUT"

            out_count += 1

        state["counted"] = True

        print(
            f"[CROSSING] {state['event_id']} | "
            f"{state['direction']} | "
            f"Plate: {state['best_plate']}"
        )

    state["stable_side"] = current_side


# ============================================================
# MAIN LOOP
# ============================================================

print("[RUNNING] Processing video...")

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        # ----------------------------------------------------
        # EXPIRE OLD EVENTS
        # ----------------------------------------------------

        expired_events = []

        for event_id, state in events.items():

            frames_missing = (
                total_frames - state["last_seen_frame"]
            )

            if frames_missing > retention_frames:

                expired_events.append(event_id)

        for event_id in expired_events:

            events.pop(
                event_id,
                None
            )

            expired_raw_ids = [

                raw_id

                for raw_id, mapped_event in raw_to_event.items()

                if mapped_event == event_id
            ]

            for raw_id in expired_raw_ids:

                raw_to_event.pop(
                    raw_id,
                    None
                )

                raw_id_last_seen.pop(
                    raw_id,
                    None
                )

            # Only OCR memory is cleared.
            # The CSV row is not deleted.
            plate_reader.clear_track(event_id)

            print(
                f"[EVENT EXPIRED] {event_id} | "
                f"CSV record preserved"
            )

        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            source=frame,
            conf=DETECTION_CONFIDENCE,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        processed_raw_ids = set()

        if results and results[0].boxes is not None:

            boxes = results[0].boxes

            if (
                boxes.id is not None
                and len(boxes) > 0
            ):

                coordinates = (
                    boxes.xyxy.cpu().numpy()
                )

                track_ids = (
                    boxes.id.int().cpu().tolist()
                )

                confidences = (
                    boxes.conf.cpu().tolist()
                    if boxes.conf is not None
                    else [0.0] * len(track_ids)
                )

                # ------------------------------------------------
                # PROCESS EACH DETECTED PLATE
                # ------------------------------------------------

                for box_values, raw_id, detection_conf in zip(
                    coordinates,
                    track_ids,
                    confidences
                ):

                    raw_id = int(raw_id)

                    if raw_id in processed_raw_ids:
                        continue

                    processed_raw_ids.add(raw_id)

                    x1, y1, x2, y2 = map(
                        int,
                        box_values
                    )

                    x1 = max(0, min(x1, frame_width - 1))
                    y1 = max(0, min(y1, frame_height - 1))

                    x2 = max(0, min(x2, frame_width))
                    y2 = max(0, min(y2, frame_height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    box = (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    )

                    # ------------------------------------------------
                    # CHECK WHETHER BYTETRACK ID WAS REUSED
                    # ------------------------------------------------

                    event_id = raw_to_event.get(raw_id)

                    state = None

                    if event_id in events:

                        last_seen = raw_id_last_seen.get(
                            raw_id,
                            events[event_id]["last_seen_frame"]
                        )

                        missing_frames = (
                            total_frames - last_seen
                        )

                        if (
                            missing_frames
                            > raw_id_reuse_gap_frames
                        ):

                            print(
                                f"[ID REUSED] ByteTrack ID {raw_id} | "
                                f"Previous event: {event_id} | "
                                f"Creating a new event"
                            )

                            state = None

                        else:

                            state = events[event_id]

                    # ------------------------------------------------
                    # CREATE A FRESH EVENT WHEN NECESSARY
                    # ------------------------------------------------

                    if state is None:

                        state = create_event(
                            raw_id,
                            box,
                            total_frames
                        )

                        event_id = state["event_id"]

                    else:

                        raw_id_last_seen[raw_id] = total_frames

                    # ------------------------------------------------
                    # UPDATE MOVEMENT
                    # ------------------------------------------------

                    update_event_position(
                        state,
                        box,
                        total_frames
                    )

                    center = state["last_center"]

                    # ------------------------------------------------
                    # OCR BEFORE AND AFTER CROSSING
                    # ------------------------------------------------

                    interval = (
                        OCR_VERIFY_INTERVAL
                        if state["best_plate"] != "SCANNING"
                        else OCR_RETRY_INTERVAL
                    )

                    if (
                        total_frames - state["last_ocr_frame"]
                        >= interval
                    ):

                        state["last_ocr_frame"] = total_frames

                        plate_reader.extract_plate_number(
                            frame,
                            box,
                            event_id
                        )

                        update_best_plate(state)

                    update_best_plate(state)

                    # ------------------------------------------------
                    # CHECK TRIPWIRE
                    # ------------------------------------------------

                    check_crossing(
                        state,
                        center[1]
                    )

                    # ------------------------------------------------
                    # CSV EVENT
                    # ------------------------------------------------

                    if state["counted"]:

                        logger.log_event(
                            event_id=event_id,
                            plate_text=state["best_plate"],
                            status=state["direction"],
                            confidence=max(
                                0.0,
                                state["best_plate_confidence"]
                            )
                        )

                    # ------------------------------------------------
                    # DRAW DETECTION
                    # ------------------------------------------------

                    color = (
                        (0, 255, 0)
                        if state["counted"]
                        else (255, 180, 0)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = (
                        f"{event_id} | "
                        f"{state['best_plate']} | "
                        f"{detection_conf:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        # ----------------------------------------------------
        # DRAW TRIPWIRE AND COUNTERS
        # ----------------------------------------------------

        cv2.line(
            frame,
            (0, line_y),
            (frame_width, line_y),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"IN: {in_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"OUT: {out_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        writer.write(frame)

        cv2.imshow(
            "Number Plate Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print("\n[COMPLETED] Processing finished.")

    print(f"Total frames: {total_frames}")
    print(f"IN count: {in_count}")
    print(f"OUT count: {out_count}")

    print(f"CSV saved: {CSV_PATH}")
    print(f"Video saved: {OUTPUT_PATH}")