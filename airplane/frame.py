from roboflow import Roboflow
rf = Roboflow(api_key="4g1R6Gj4zE29SSNKROrm")
project = rf.workspace("mia-pqqxd").project("frame-y9ksc")
version = project.version(1)
model = project.version(1).model
dataset = version.download("yolov8")
