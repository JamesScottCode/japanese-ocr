import os
import numpy as np

class DataLoader:
    def __init__(self, file_path, record_size=2952, metadata_size=216, image_size=2736, width=72, height=76):
        self.file_path = file_path
        self.record_size = record_size
        self.metadata_size = metadata_size
        self.image_size = image_size
        self.width = width
        self.height = height

    # read one record and return its metadata and reshaped image from 4-bit values.
    def _read_record(self, f, record_index):
        f.seek(record_index * self.record_size)
        record = f.read(self.record_size)
        if len(record) != self.record_size:
            return None, None
        metadata = record[:self.metadata_size]
        raw_image = record[self.metadata_size:self.metadata_size + self.image_size]
        # Convert each byte to two 4-bit values
        img_4bit = np.zeros(self.width * self.height, dtype=np.uint8)
        for i, byte_val in enumerate(raw_image):
            img_4bit[2 * i] = (byte_val >> 4) & 0x0F
            img_4bit[2 * i + 1] = byte_val & 0x0F
        return metadata, img_4bit.reshape((self.height, self.width))

    # load multiple records from file, converting them to images with corresponding class labels.
    def load(self, num_records=10608, records_per_class=208):
        file_size = os.path.getsize(self.file_path)
        total_records = file_size // self.record_size
        num_records = min(num_records, total_records) if num_records else total_records
        images, labels = [], []
        with open(self.file_path, 'rb') as f:
            for idx in range(num_records):
                metadata, img = self._read_record(f, idx)
                if img is None:
                    continue
                label = idx // records_per_class
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)