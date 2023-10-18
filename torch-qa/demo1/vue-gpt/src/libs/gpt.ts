import jwtApi from "./jwtApi";

const API_URL = import.meta.env.VITE_API_URL;

export interface ChatMessage {
   id: number;
   role: "user" | "assistant" | "system";
   content: string;
}

export interface IGetLastConversationMessagesResp {
   conversationId: number;
   messages: ChatMessage[];
}

export interface IAskReq {
   conversationId: number;
   content: string;
}


const decoder = new TextDecoder("utf-8");

type GeneratingFn = (content: string) => void;

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

   async sendChatMessageStream(content: string, generatingFn: GeneratingFn): Promise<ChatMessage> {
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
         return await this.readStream(reader, status, generatingFn);
      }
      return this.getLastMessage();
   }

   async readStream(
      reader: ReadableStreamDefaultReader<Uint8Array>,
      status: number,
      generatingFn: GeneratingFn | undefined = undefined
   ): Promise<ChatMessage> {
      let partialLine = "";

      while (status > 0) {
         const { value, done } = await reader.read();
         if (done) {
            console.log('DONE');
            break;
         }

         const decodedText = decoder.decode(value, { stream: true });
         console.log(`decoded ${decodedText}`, status);

         if (status !== 200) {
            const error = JSON.parse(decodedText);
            //const content = json.error.message ?? decodedText;
            throw new Error(error);
         }

         const chunk = partialLine + decodedText;
         const newLines = chunk.split(/\r?\n/);
         partialLine = newLines.pop() ?? "";
         for (const line of newLines) {
            if (line.length === 0) continue; // ignore empty message
            if (line.startsWith(":")) continue; // ignore sse comment message
            //console.log("line='" + line + "'")
            if (line === "data: [DONE]") {
               //console.log("END");
               return this.getLastMessage();
            }

            const content = JSON.parse(line);
            generatingFn?.call(generatingFn, content);
            //console.log("appendLastMessageContent1", content)
            //this.appendLastMessageContent(content);
            //console.log("appendLastMessageContent2", content)
         }
      }

      return this.getLastMessage();
   }

   appendLastMessageContent(content: string) {
      this.messageList[this.messageList.length - 1].content += content;
   }

   getLastMessage() {
      return this.messageList[this.messageList.length - 1];
   }

   async ask(req: IAskReq, generatingFn: GeneratingFn): Promise<ChatMessage> {
      const { body, status } = await this.postConversation(req);
      const reader = body!.getReader();
      return await this.readStream(reader, status, generatingFn);
   }

   async postChat(messageList: ChatMessage[]) {
      const accessToken = localStorage.getItem('accessToken');
      //const result = await fetch("/api/v1/chat/completions", {
      const result = await fetch("http://127.0.0.1:5000/api/v1/chat/stream", {
         method: "post",
         // signal: AbortSignal.timeout(8000),
         headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
         },
         body: JSON.stringify({
            messages: messageList,
         }),
      });
      return result;
   }

   getLastConversationMessages() {
      return jwtApi.post<IGetLastConversationMessagesResp>("/api/v1/chat/getLastConversation");
   }

   async postConversation(req: IAskReq) {
      const accessToken = localStorage.getItem('accessToken');
      const result = await fetch(`${API_URL}/api/v1/chat/conversation`, {
         method: "post",
         // signal: AbortSignal.timeout(8000),
         headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
         },
         body: JSON.stringify({
            conversationId: req.conversationId,
            content: req.content,
         }),
      });
      return result;
   }
}