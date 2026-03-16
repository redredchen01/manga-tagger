import http.server
import socketserver
import webbrowser
import time

# Determine the port
PORT = 8000


# Create a custom handler
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        if self.path == "/health":
            self.wfile.write(b'{"status":"healthy","message":"Test server working"}')
        else:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Manga Tagger Server</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                    .success { color: green; font-size: 24px; }
                    .info { color: blue; font-size: 18px; margin: 20px; }
                </style>
            </head>
            <body>
                <div class="success">🎉 MANGA TAGGER SERVER IS RUNNING!</div>
                <div class="info">
                    <p>✅ Server successfully started on localhost</p>
                    <p>✅ Port: 8000</p>
                    <p>✅ Status: Working</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())


# Try to start the server
try:
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Server running on http://localhost:{PORT}")
        print(f"Also accessible at: http://127.0.0.1:{PORT}")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to exit...")
