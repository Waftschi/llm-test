"""Web UI for the barkeeper:  python3 server.py [simple|simple-en] [port]  ->  http://localhost:8000

Tiny JSON API on top of barkeeper_core.py (standard library only) + serves web/index.html.
  GET  /api/params   model + training parameters
  GET  /api/data     training examples, right/wrong
  GET  /api/compare  one matrix vs two layers
  POST /api/chat     {sid, text}    one line of chat (also: falsch, reset, "sentence = answer")
  POST /api/teach    {sid, intent}  answer to "what should I have done?"
  POST /api/skip     {sid}          teacher says nothing
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from barkeeper_core import Barkeeper, Session, HERE

args = sys.argv[1:]
PORT = next((int(a) for a in args if a.isdigit()), 8000)
PROFILE = os.environ.get("BARKEEPER_PROFILE") or next((a for a in args if a.startswith("simple")), "")

bar = Barkeeper(PROFILE)
sessions = {}                                   # one Session per browser tab

def session(sid):
    return sessions.setdefault(sid or "default", Session())

class Handler(BaseHTTPRequestHandler):
    def send(self, body, status=200, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        routes = {"/api/params": bar.params, "/api/data": bar.data, "/api/compare": bar.compare}
        if self.path in routes:
            return self.send(routes[self.path]())
        if self.path in ("/", "/index.html"):
            return self.send(open(os.path.join(HERE, "web", "index.html"), "rb").read(),
                             ctype="text/html; charset=utf-8")
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.send({"error": "bad json"}, 400)
        s = session(body.get("sid"))
        if self.path == "/api/chat":
            text = str(body.get("text", "")).strip()
            return self.send(bar.handle(s, text) if text else {"type": "error", "message": "say something"})
        if self.path == "/api/teach":
            intent = str(body.get("intent", "")).strip()
            return self.send(bar.teach(s, intent) if intent else {"type": "error", "message": "no answer given"})
        if self.path == "/api/skip":
            s.pending = None
            return self.send({"type": "skipped"})
        self.send({"error": "not found"}, 404)

    def log_message(self, fmt, *a):             # quiet
        pass

if __name__ == "__main__":
    print(f"🍺 barkeeper web UI on http://localhost:{PORT}  (profile: {PROFILE or 'default'})")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
