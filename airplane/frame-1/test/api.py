from roboflow import Roboflow
rf = Roboflow(api_key="4g1R6Gj4zE29SSNKROrm")
project = rf.workspace("mia-pqqxd").project("frame-y9ksc")
model = project.version(1).model

print(model.predict(r"C:\Users\edayu\PycharmProjects\Dental\yolov8\frame-1\test\images\frame_21110_jpg.rf.170f09e4fec435be95940fdfed11f454.jpg", confidence= 40, overlap= 30).save("prediction.jpg"))