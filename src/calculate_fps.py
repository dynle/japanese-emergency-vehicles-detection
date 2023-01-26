import time
import cv2

#video capture object
cap=cv2.VideoCapture(1) #iphone camera

fps_start_time = 0
fps = 0

# capture the frames..
while True:
    ret, frame = cap.read()

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/time_diff
    fps_start_time = fps_end_time

    fps_text = f"FPS: {fps}"

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('Frame', frame)  # Display the resulting frame
    key = cv2.waitKey(1)
    if key == 27:  # click esc key to exit
        break

cap.release()
cv2.destroyAllWindows()