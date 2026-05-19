package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OCRRequest struct {
	Config *Config
	PNG    []byte
}

type chatRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	KeepAlive *int          `json:"keep_alive,omitempty"`
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content []contentPart `json:"content"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func RunOCR(req OCRRequest) (string, error) {
	payload, err := buildOCRPayload(req)
	if err != nil {
		return "", err
	}
	body, err := postOCR(req.Config.OCREndpoint, payload)
	if err != nil {
		return "", err
	}
	return parseOCRResponse(body)
}

func buildOCRPayload(req OCRRequest) ([]byte, error) {
	encoded := base64.StdEncoding.EncodeToString(req.PNG)
	keepAliveForever := -1
	payload := chatRequest{
		Model: req.Config.OCRModel,
		Messages: []chatMessage{
			{
				Role: "user",
				Content: []contentPart{
					{Type: "text", Text: req.Config.OCRPrompt},
					{Type: "image_url", ImageURL: &imageURL{
						URL: "data:image/png;base64," + encoded,
					}},
				},
			},
		},
		KeepAlive: &keepAliveForever,
	}
	return json.Marshal(payload)
}

func postOCR(endpoint string, payload []byte) ([]byte, error) {
	client := &http.Client{Timeout: 300 * time.Second}
	req, err := http.NewRequest("POST", endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("ocr endpoint %d: %s", resp.StatusCode, truncate(string(body), 200))
	}
	return body, nil
}

func parseOCRResponse(data []byte) (string, error) {
	var r chatResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return "", fmt.Errorf("invalid ocr response: %w", err)
	}
	if r.Error != nil {
		return "", fmt.Errorf("ocr error: %s", r.Error.Message)
	}
	if len(r.Choices) == 0 {
		return "", fmt.Errorf("ocr response has no choices")
	}
	return r.Choices[0].Message.Content, nil
}

func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}
