# _Virtual Mouse using Mediapipe._

### About/Overview:
✨Virtual mouse is a cool, fun, experimental project which tracks hands gestures  using the Google's 
mediapipe library and controls the cursor on the screen.✨
### Features:
- User can see a live visualization of 21 3D landmarks on their hands.
- User can see the Active Range region (a Bounding box) which activates the cursor movement. Any
  hand outside this region will not trigger the cursor.
- User can trigger mouse left click by folding their palms.

### How to Run:
```sh
python3 virtualmouse.py # for python3 mouse control.
python3 HandTracking.py #for python3 Hand tracking.

python virtualmouse.py # for python2 mouse control
python HandTracking.py #for python2 Hand tracking.
```
### How to use the Program:
- Firstly, clone this repo to your desktop. 
```sh
git clone https://github.com/aadityasp/Virtual_Mouse.git
```
- Next install all the prerequisite packages using 
 ```sh
 pip install -r requirements.txt
```

- Next run the virtualmouse.py to control the mouse using hand.
- If you only want to Track the hands without controlling the mouse, run only the HandTracking.py file.

### Limitations:
- The Gestures work only based on angles between various keypoints on the palm. It
can be further improved by using a gesture recognition algorithm by training a DL model.

## License

MIT

**Free Software, Hell Yeah!**
