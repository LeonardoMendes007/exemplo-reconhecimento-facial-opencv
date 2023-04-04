using OpenCvSharp;

using (var videoCapture = new VideoCapture("Resources/video.mp4"))
{
    var classifier = new CascadeClassifier("Resources/haarcascade_frontalface_default.xml");

    var window = new Window("Face Detection");

    var frame = new Mat();

    while (true)
    {
        videoCapture.Read(frame);

        if (frame.Empty())
        {
            break;
        }

        var gray = new Mat();
        Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

        var faces = classifier.DetectMultiScale(gray);

        foreach (var face in faces)
        {
            Cv2.Rectangle(frame, face, Scalar.Red, 2);
        }

        window.ShowImage(frame);
        int key = Cv2.WaitKeyEx(1);
        if (key == 'q')
        {
            break;
        }
    }
}

