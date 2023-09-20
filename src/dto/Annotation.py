class WordAnnotation:
    def __init__(self, polygon, text):
        self.polygon = polygon
        self.text = text

    @classmethod
    def from_dict(cls, data):
        return cls(polygon=data["polygon"], text=data["text"])

    def to_dict(self):
        return {"polygon": self.polygon, "text": self.text}


class Annotation:
    def __init__(self, polygon, text, label, words, linking, annotation_id):
        self.polygon = polygon
        self.text = text
        self.label = label
        self.words = [WordAnnotation.from_dict(word) for word in words]
        self.linking = linking
        self.id = annotation_id

    @classmethod
    def from_dict(cls, data):
        return cls(
            polygon=data["polygon"],
            text=data["text"],
            label=data["label"],
            words=data["words"],
            linking=data["linking"],
            annotation_id=data["id"]
        )

    def to_dict(self):
        return {
            "polygon": self.polygon,
            "text": self.text,
            "label": self.label,
            "words": [word.to_dict() for word in self.words],
            "linking": self.linking,
            "id": self.id
        }