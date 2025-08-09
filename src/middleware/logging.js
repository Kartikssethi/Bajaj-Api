const loggingMiddleware = (req, res, next) => {
  // Check if content-type is json AND there's no file being uploaded
  // This helps prevent express.json() from trying to parse multipart/form-data wrongly
  const isJsonRequest =
    req.headers["content-type"]?.includes("application/json");
  const isMultipartForm = req.headers["content-type"]?.includes(
    "multipart/form-data"
  );

  if (req.method === "POST" && isJsonRequest && !isMultipartForm) {
    const timestamp = new Date().toISOString();
    const requestId = `req_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    console.log(
      `[${requestId}] ${timestamp} - Incoming ${req.method} ${req.path}`
    );
    // Log body only if it's expected to be JSON and parsed by express.json()
    console.log(
      `[${requestId}] Request Body:`,
      JSON.stringify(req.body, null, 2)
    );
    req.requestId = requestId;
  } else if (req.method === "POST" && isMultipartForm) {
    // Log for multipart requests, req.body won't be parsed yet by multer at this stage
    const timestamp = new Date().toISOString();
    const requestId = `req_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;
    console.log(
      `[${requestId}] ${timestamp} - Incoming ${req.method} ${req.path} (Multipart Request)`
    );
    req.requestId = requestId;
  }
  next();
};

module.exports = loggingMiddleware;
