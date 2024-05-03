import math
import multiprocessing as mp
import sys
from deepface import DeepFace
from os import path, environ
from typing import Any

import cv2
from shapely import Polygon, is_prepared, prepare, box


class Entity:

    def __init__(self, geometry: Polygon, index: int, name: str):
        self.geometry = geometry
        if not is_prepared(geometry):
            prepare(geometry)
        self.id = index
        self.name = name
        self.centroid = geometry.centroid
        self.vector = (0, 0)

    def update_location(self, area):
        geometry = area_to_polygon(area)
        centroid = geometry.centroid
        self.vector = math.dist(self.centroid, centroid)
        self.geometry = geometry
        self.centroid = centroid


list_of_faces = {}
idx = 0


def area_to_polygon(area):
    x = area.get('x')
    y = area.get('y')
    w = area.get('w')
    h = area.get('h')
    geometry = box(xmin=x, ymin=y, xmax=x + w, ymax=y + h)
    prepare(geometry)
    return geometry


def face_lookup(face, facial_area) -> list:
    found_face = []
    geom = area_to_polygon(facial_area)
    data = DeepFace.find(img_path=face, db_path="database", silent=True, enforce_detection=False)
    print(f"Found {len(data)} faces in the file.")
    for face_data in data:
        name = str(face_data.get("identity")[0]).split(sep=path.sep)[1]
        found_face.append((geom, name))
    return found_face


def face_detection(image):
    detected_aligned_face = DeepFace.extract_faces(img_path=image, detector_backend='ssd')
    return detected_aligned_face


def draw_outline(image, area):
    x = area.get('x')
    y = area.get('y')
    w = area.get('w')
    h = area.get('h')
    return cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)


def get_face(geom: Polygon) -> Entity | None:
    for current_entity in list_of_faces.values():
        if current_entity.geometry.contains(geom.centroid):
            print(f"found match for {current_entity.name}")
            current_entity.geometry = geom
            return current_entity
        else:
            print(f"{current_entity.geometry} does not intersect {geom}")
    return None


def update_faces_with(face: dict[str, Any], pool: mp.Pool):
    pool.apply_async(face_lookup,
                     args=(face.get("face").copy(), face.get("facial_area")),
                     callback=add_faces,
                     error_callback=no_face_found_error)


def add_faces(found_faces_list: list):
    for data in found_faces_list:
        geom = data[0]
        name = data[1]
        list_of_faces[name] = Entity(geometry=geom, name=name, index=idx + 1)
        print(f"list size = {len(list_of_faces)} added {name} with geom: {geom}")


def outline_face(image, face):
    area = face.get('facial_area')
    draw_outline(image, area)


def show_image(image):
    cv2.imshow('frame', image)


def no_face_found_error(error):
    print(f"No Face Found! {error}")


def outline_known_faces(frame):
    for current_entity in list_of_faces.values():
        bounds = current_entity.geometry.bounds
        cv2.rectangle(frame, (int(bounds[0]), int(bounds[1])), (int(bounds[2]), int(bounds[3])), (0, 0, 255), 2)


def main():
    face_lookup_pool = mp.Pool(4)
    try:
        vid = cv2.VideoCapture(0)
        detector_rate = 0
        while True:
            ret, frame = vid.read()
            try:
                face_data = face_detection(frame)
                if len(face_data) > 0:
                    for face in face_data:
                        if face.get('confidence') > 0.75:
                            geom = area_to_polygon(face.get('facial_area'))
                            known_face = get_face(geom)
                            if known_face is not None:
                                list_of_faces[known_face.name].geometry = geom
                            elif detector_rate % 10 == 0:
                                update_faces_with(face, face_lookup_pool)
                                detector_rate += 1
                            outline_face(frame, face)
                outline_known_faces(frame)
            except ValueError:
                pass
            show_image(frame)
            # the 'q' button is set as the release
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()
    except ValueError as ve:
        return str(ve)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# result = DeepFace.verify(img1_path="IMG_1947.jpeg", img2_path="IMG_2200.jpeg")
# print(json.dumps(result, indent=2))

# embedding_objs = DeepFace.represent(img_path="IMG_1947.jpeg")
# embedding = embedding_objs[0]["embedding"]

# print(json.dumps(embedding_objs, indent=2))

# objs = DeepFace.analyze(img_path="IMG_2200.jpeg",
#                         actions=['age', 'gender', 'race', 'emotion']
#                         )

# print(json.dumps(objs, indent=2))
