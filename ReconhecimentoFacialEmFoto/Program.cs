using OpenCvSharp;

var classifier = new CascadeClassifier("Resources/haarcascade_frontalface_default.xml");

var window = new Window("Face Detection");

var frame = Cv2.ImRead("Resources/foto_1.jpg");

var gray = new Mat();
Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

var faces = classifier.DetectMultiScale(gray);

foreach (var face in faces)
{
    Cv2.Rectangle(frame, face, Scalar.Red, 2);
}

while (true)
{
    window.ShowImage(frame);
    int key = Cv2.WaitKeyEx(1);
    if(key == 'q')
    {
        break;
    }
}

window.Close();