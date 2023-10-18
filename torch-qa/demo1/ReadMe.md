```mermaid
sequenceDiagram
    Client ->> Server: /api/v1/chat/GetLastConversationMessages;
    Server ->> +GptService: getLastConversationMessages(loginName);
    GptService ->> +GptRepo: getConversation(loginName);
    GptRepo ->> +Database: query Conversations; Database -->> -GptRepo: ConversationEntity ;
    GptService ->> GptRepo: getConversationDetails(id);
    GptRepo ->> +Database: query ConversationDetails ; Database -->> -GptRepo: list[ConversationDetailEntity] ;
    GptService -->> -Server: list[GptMessage];
    Server -->> Client: { messages: GptMessage[] };
```

send query
```mermaid
sequenceDiagram
    Client ->> Server: /api/v1/chat/stream <br> { conversationId: number, message: string };
    Server ->> +GptService: stream <br> (loginName, conversationId, message);
    GptService ->> +GptRepo: getConversation(loginName);
    GptRepo ->> +Database: query ; Database -->> -GptRepo: ConversationEntity ;
    GptService ->> GptRepo: getConversationDetails(id);
    GptRepo ->> +Database: query ConversationDetails ; Database -->> -GptRepo: list[ConversationDetailEntity] ;
    
    GptService -->> -Server: list[GptMessage];
    Server -->> Client: stream ;
```