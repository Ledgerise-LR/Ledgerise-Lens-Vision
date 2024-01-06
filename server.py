from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
from detect import processImage
from getBlur import getBlur

hostName = "https://lens-ledgerise.onrender.com"
serverPort = 8080


def process_image(image_bytes):
    result = processImage(image_bytes)
    return json.dumps(result)


def get_blur(tokenUri):
    result = getBlur(tokenUri)
    return json.dumps(result)


class MyServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/real-time":
            content_type, params = cgi.parse_header(self.headers["content-type"])

            if content_type == "application/json":
                content_length = int(self.headers["content-length"])
                post_data = self.rfile.read(content_length)
                post_data_dict = json.loads(post_data.decode("utf-8"))

                if "image" in post_data_dict:
                    base64_image = post_data_dict["image"]
                    result_json = process_image(base64_image)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(result_json).encode("utf-8"))
                    return
        elif self.path == "/privacy/blur":
            content_type, params = cgi.parse_header(self.headers["content-type"])

            if content_type == "application/json":
                content_length = int(self.headers["content-length"])
                post_data = self.rfile.read(content_length)
                post_data_dict = json.loads(post_data.decode("utf-8"))

                if "tokenUri" in post_data_dict:
                    tokenUri = post_data_dict["tokenUri"]
                    result_json = get_blur(tokenUri)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(result_json).encode("utf-8"))
                    return

        self.send_response(400)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Invalid request"}).encode("utf-8"))


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
