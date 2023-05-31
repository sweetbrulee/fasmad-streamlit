from service.kernel.facialrecognition.Dlib_face_recognition_from_camera.my_face_reco import (
    Face_Recognizer,
)
from service.kernel.facialrecognition.Dlib_face_recognition_from_camera import (
    my_features_extraction_to_csv,
)
from .kernel.firedetection.interface import start_fire_detect2
from .message_queue import MetadataQueueService


class InferenceService:
    def __init__(self):
        # singleton class, cannot be initialized
        raise NotImplementedError("This is a singleton class, cannot be initialized")


class FireDetection(InferenceService):
    @classmethod
    def create(cls, frame):
        return start_fire_detect2(frame)


class FaceIdentification(InferenceService):
    face_recognizer = Face_Recognizer()

    @classmethod
    def create(cls, frame):
        return cls.face_recognizer.run(frame)

    @classmethod
    def register(cls, name, faces: list):
        my_features_extraction_to_csv.main(faces, name)
