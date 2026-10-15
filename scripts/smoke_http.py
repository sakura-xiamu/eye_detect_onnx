import io
import json
import urllib.request
from PIL import Image

status = json.load(urllib.request.urlopen("http://127.0.0.1:8001/api/gaze/status"))
print("status", status["initialized"], status["provider"], status["error"])

buffer = io.BytesIO()
Image.new("RGB", (64, 64), (0, 0, 0)).save(buffer, format="JPEG")
boundary = "----eyegaze"
chunks = []
fields = {
    "pitch_range": "-30,5",
    "yaw_range": "-22,22",
    "source": "image",
}
for name, value in fields.items():
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    )
chunks.append(
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"blank.jpg\"\r\n"
    f"Content-Type: image/jpeg\r\n\r\n".encode()
)
chunks.append(buffer.getvalue())
chunks.append(f"\r\n--{boundary}--\r\n".encode())
body = b"".join(chunks)
request = urllib.request.Request(
    "http://127.0.0.1:8001/api/gaze/image",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
payload = json.load(urllib.request.urlopen(request))
print(payload["result"]["message"], payload["result"]["looking_outside"], payload["provider"])

bad = body.replace(b"-30,5", b"5,-30")
bad_request = urllib.request.Request(
    "http://127.0.0.1:8001/api/gaze/image",
    data=bad,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
try:
    urllib.request.urlopen(bad_request)
    print("bad-window-unexpected-success")
except urllib.error.HTTPError as exc:
    print("bad-window", exc.code)
