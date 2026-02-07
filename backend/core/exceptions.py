class BaseAPIException(Exception):
    status_code: int = 500
    detail: str = "Server Error"
    
    def __init__(self, detail: str = None, status_code: int = None):
        if detail:
            self.detail = detail
        if status_code:
            self.status_code = status_code
        super().__init__(self.detail)

class BadRequestException(BaseAPIException):
    status_code = 400
    detail = "Bad Request"

class NotFoundException(BaseAPIException):
    status_code = 404
    detail = "Not Found"

class ServerException(BaseAPIException):
    status_code = 500
    detail = "Internal Server Error"
