from livereload import Server

server = Server()
server.watch('index.html')
server.watch('static/')
server.serve(root='.', port=8000)

