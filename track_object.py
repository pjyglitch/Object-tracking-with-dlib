# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-l", "--label", required=True,
	help="class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(
    args["prototxt"], args["model"]
)

# initialize the video stream, dlib correlation tracker, output video
# writer, and predicted class label
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
tracker = None
writer = None
label = ""

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the next frame from the video file
	(grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
	if frame is None:
		break

	# resize the frame for faster processing and then convert the
	# frame from BGR to RGB ordering (dlib needs RGB ordering)
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)


		if tracker is None:
		blob = cv2.dnn.blobFromImage(
			frame, 0.007843, (300, 300), 127.5
		)
		net.setInput(blob)
		detections = net.forward()

		(h, w) = frame.shape[:2]

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence < args["confidence"]:
				continue

			idx = int(detections[0, 0, i, 1])
			if CLASSES[idx] != args["label"]:
				continue

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x1, y1, x2, y2) = box.astype("int")

			x1, y1 = max(0, x1), max(0, y1)
			x2, y2 = min(w - 1, x2), min(h - 1, y2)

			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(x1, y1, x2, y2)
			tracker.start_track(rgb, rect)

			label = args["label"]
			break


	else:
		pass
    
	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
# check to see if we need to release the video writer pointer
# do a bit of cleanup
fps.stop()
print("Elapsed time: {:.2f} seconds".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()

cv2.destroyAllWindows()
vs.release()
