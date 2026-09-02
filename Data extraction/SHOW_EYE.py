"""
Show eye regions and pupil detection on ET frame.
Usage: python show_eye.py --image "E:\aria_work\sree_dt\with_data\enhanced_000694.jpg"
"""
import cv2
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W  = gray.shape[:2]

    # Eye regions from click data
    LEFT_EYE  = (46, 67,  280, 233)  # x1,y1,x2,y2
    RIGHT_EYE = (320+17, 102, 320+249, 237)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for (x1,y1,x2,y2), label, color in [
        (LEFT_EYE,  "LEFT",  (0,255,0)),
        (RIGHT_EYE, "RIGHT", (0,200,255))
    ]:
        eye_img = gray[y1:y2, x1:x2]
        h_e, w_e = eye_img.shape

        # Draw eye boundary
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 1)
        cv2.putText(vis, label, (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Pupil detection — threshold 30, most circular
        blurred  = cv2.GaussianBlur(eye_img, (5,5), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        thresh   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = [c for c in cnts
                 if 500 < cv2.contourArea(c) < 5000
                 and 4*np.pi*cv2.contourArea(c)/(cv2.arcLength(c,True)**2+1e-8) > 0.10]

        if valid:
            best  = max(valid, key=lambda c: 4*np.pi*cv2.contourArea(c)/(cv2.arcLength(c,True)**2+1e-8))
            area  = cv2.contourArea(best)
            circ  = 4*np.pi*area/(cv2.arcLength(best,True)**2+1e-8)
            M     = cv2.moments(best)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00']) + x1
                cy = int(M['m01']/M['m00']) + y1
                r  = int(np.sqrt(area/np.pi))
                cv2.circle(vis, (cx,cy), r,   (0,255,255), 2)
                cv2.circle(vis, (cx,cy), 3,   (0,255,255), -1)
                diam_mm = (2*r) / 9.0
                print(f"[{label}] pupil center=({cx},{cy}) r={r}px diam={diam_mm:.2f}mm circ={circ:.3f}")
        else:
            print(f"[{label}] NO PUPIL FOUND")

    cv2.namedWindow("Eye + Pupil", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Eye + Pupil", vis)
    print("\nPress Q to quit.")
    while True:
        if cv2.waitKey(100) & 0xFF in [ord('q'), ord('Q'), 27]:
            break
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()