export interface ChatMessage {
   id: number;
   role: "user" | "assistant" | "system";
   content: string;
}

const decoder = new TextDecoder("utf-8");

export class ChatGpt {
   apiKey: string = "";
   messageList: ChatMessage[] = [];

   async sendChatMessage(content: string): Promise<ChatMessage> {
      this.messageList.push({
         id: this.messageList.length + 1,
         role: 'user',
         content: content
      });
      const { body, status } = await this.postChat(this.messageList);
      this.messageList.push({
         id: this.messageList.length + 1,
         role: 'assistant',
         content: ''
      });
      if (body) {
         const reader = body.getReader();
         return await this.readStream(reader, status);
      }
      return this.getLastMessage();
   }

   async readStream(
      reader: ReadableStreamDefaultReader<Uint8Array>,
      status: number
   ): Promise<ChatMessage> {
      //let partialLine = "";

      while (status > 0) {
         const { value, done } = await reader.read();
         if (done) break;

         const decodedText = decoder.decode(value, { stream: true });
         this.appendLastMessageContent(decodedText);

         // if (status !== 200) {
         //    const json = JSON.parse(decodedText); // start with "data: "
         //    const content = json.error.message ?? decodedText;
         //    this.appendLastMessageContent(content);
         //    return this.getLastMessage();
         // }

         //const chunk = partialLine + decodedText;

         // const newLines = chunk.split(/\r?\n/);
         // partialLine = newLines.pop() ?? "";
         // for (const line of newLines) {
         //    if (line.length === 0) continue; // ignore empty message
         //    if (line.startsWith(":")) continue; // ignore sse comment message
         //    if (line === "data: [DONE]") {
         //       return this.getLastMessage();
         //    }

         //    // const json = JSON.parse(line.substring(6)); // start with "data: "
         //    // const content =
         //    //    status === 200
         //    //       ? json.choices[0].delta.content ?? ""
         //    //       : json.error.message;
         //    this.appendLastMessageContent(line);
         // }
      }

      return this.getLastMessage();
   }

   appendLastMessageContent(content: string) {
      this.messageList[this.messageList.length - 1].content += content;
   }

   getLastMessage() {
      return this.messageList[this.messageList.length - 1];
   }

   async postChat(messageList: ChatMessage[]) {
      //const result = await fetch("/api/v1/chat/completions", {
      const result = await fetch("http://127.0.0.1:5000/api/v1/chat/stream", {
         method: "post",
         // signal: AbortSignal.timeout(8000),
         headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.apiKey}`,
         },
         body: JSON.stringify({
            messages: messageList,
         }),
      });
      return result;
   }
}