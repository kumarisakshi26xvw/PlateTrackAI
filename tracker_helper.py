import csv
import os
from datetime import datetime


class VehicleLogger:

    COLUMNS = [
        "License Plate",
        "Status",
        "Timestamp"
    ]

    def __init__(self, filename="vehicle_log.csv"):

        self.filename = filename
        self.rows = []

        # Event ID -> CSV row index
        self.logged_events = {}

        # Event ID -> highest accepted OCR confidence
        self.best_confidences = {}

        self._load_existing_rows()

    def _load_existing_rows(self):

        if not os.path.exists(self.filename):
            return

        try:
            with open(
                self.filename,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                reader = csv.DictReader(file)

                for source_row in reader:

                    row = {
                        column: source_row.get(column, "") or ""
                        for column in self.COLUMNS
                    }

                    self.rows.append(row)

        except (OSError, csv.Error) as error:

            print(f"[CSV WARNING] {error}")
            self.rows = []

    def log_event(
        self,
        event_id,
        plate_text,
        status,
        confidence=0.0
    ):

        if status not in ("IN", "OUT"):
            return

        event_id = str(event_id)

        plate_text = str(
            plate_text or "SCANNING"
        ).strip().upper()

        if not plate_text:
            plate_text = "SCANNING"

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        # A SCANNING value is never an accepted plate.
        accepted = plate_text != "SCANNING"

        # Create exactly one row for this event.
        if event_id not in self.logged_events:

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            row_index = len(self.rows)

            self.rows.append({
                "License Plate": (
                    plate_text if accepted else "SCANNING"
                ),
                "Status": status,
                "Timestamp": timestamp
            })

            self.logged_events[event_id] = row_index

            self.best_confidences[event_id] = (
                confidence if accepted else -1.0
            )

            self._save()

            print(
                f"[CSV CREATED] Event: {event_id} | "
                f"Plate: {plate_text} | "
                f"Status: {status} | "
                f"Confidence: {confidence:.4f}"
            )

            return

        # Update the existing row, never create another one.
        row_index = self.logged_events[event_id]

        if not accepted:
            return

        previous_confidence = self.best_confidences.get(
            event_id,
            -1.0
        )

        if confidence <= previous_confidence:
            return

        previous_plate = self.rows[row_index]["License Plate"]

        self.best_confidences[event_id] = confidence

        if previous_plate != plate_text:

            self.rows[row_index]["License Plate"] = plate_text

            # Status and timestamp are deliberately untouched.
            self._save()

            print(
                f"[CSV UPDATED] Event: {event_id} | "
                f"{previous_plate} -> {plate_text} | "
                f"Confidence: {confidence:.4f}"
            )

        else:

            print(
                f"[CSV BEST CONFIDENCE] Event: {event_id} | "
                f"Confidence: {confidence:.4f}"
            )

    def _save(self):

        directory = os.path.dirname(
            os.path.abspath(self.filename)
        )

        os.makedirs(directory, exist_ok=True)

        temporary_path = self.filename + ".tmp"

        try:

            with open(
                temporary_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=self.COLUMNS
                )

                writer.writeheader()
                writer.writerows(self.rows)

            os.replace(
                temporary_path,
                self.filename
            )

        except OSError as error:

            print(f"[CSV ERROR] Could not save CSV: {error}")

            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass