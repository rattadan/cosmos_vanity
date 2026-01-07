# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .

# Build the binary with static linking
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o cosmosvanity .

# Final stage
FROM alpine:latest

WORKDIR /root/
COPY --from=builder /app/cosmosvanity .

# Set the entrypoint
ENTRYPOINT ["./cosmosvanity"]
