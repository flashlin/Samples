

|client call API | client has Content |api implement |result
|---|---
|async |yes |async |work
|async |yes |async ConfigureAwait(false)|work
|async ConfigureAwait(false)|yes |async |**block**
|async |yes |static async ConfigureAwait(false) |work
|task1(Content), task2(Content), Task.WhenAll |yes |async |**block**


