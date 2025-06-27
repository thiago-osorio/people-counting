import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import supervision as sv
import numpy as np
import shutil
import cv2
import os
import av

def update_representative_embeddings(person, new_embedding, max_embeddings, sim_threshold):
    embeddings = person["embeddings"]
    if len(embeddings) == 0:
        embeddings.append(new_embedding)
        return True

    similarities = cosine_similarity([new_embedding], embeddings)[0]

    if np.max(similarities) > sim_threshold:
        return False

    if len(embeddings) < max_embeddings:
        embeddings.append(new_embedding)
        return True

    most_similar_idx = np.argmax(similarities)
    embeddings[most_similar_idx] = new_embedding
    return True

SIM_THRESH = 0.40
REP_THRESH = 0.50
EMB_LIMIT = 10
IMG_LIMIT = 10

people_model = YOLO("yolov8n.pt")
face_app = FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=-1, det_size=(160, 160))

people_db = []
person_id_counter = 0

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def recv(self, frame):
        global people_db, person_id_counter

        img = frame.to_ndarray(format="bgr24")
        results = people_model(img)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections)
        
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            faces = face_app.get(person_crop)
            if not faces:
                continue

            face = faces[0]
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            fx1 += x1
            fx2 += x1
            fy1 += y1
            fy2 += y1
            
            embedding = face.embedding
            
            found_match = False
            for person in people_db:
                similarities = cosine_similarity([embedding], person["embeddings"])[0]
                if np.any(similarities >= SIM_THRESH):
                    was_updated = update_representative_embeddings(person, embedding, EMB_LIMIT, REP_THRESH)
                    
                    if was_updated and person["img_count"] < IMG_LIMIT:
                        folder = f"rostos_salvos/pessoa_{person['id']:03d}"
                        os.makedirs(folder, exist_ok=True)
                        filename = os.path.join(folder, f"face_{person['img_count']:02d}.jpg")
                        face_crop = frame[fy1:fy2, fx1:fx2]
                        cv2.imwrite(filename, face_crop)
                        person["img_count"] += 1
                    
                    found_match = True
                    break
            
            if not found_match:
                new_id = person_id_counter
                person_id_counter += 1
                
                people_db.append({
                    "id": new_id,
                    "embeddings": [embedding],
                    "img_count": 1
                })
                
                folder = f"rostos_salvos/pessoa_{new_id:03d}"
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, f"face_00.jpg")
                face_crop = frame[fy1:fy2, fx1:fx2]
                cv2.imwrite(filename, face_crop)
        
        labels = [f"person {confidence:.2f}" for confidence in detections.confidence]
    
        annotated_image = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections, 
            labels=labels
        )
        
        cv2.putText(
            annotated_image,
            f"Rostos unicos: {len(people_db)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Contador de Rostos Únicos - Streamlit + YOLO + InsightFace")

webrtc_streamer(
    key="face-app",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)