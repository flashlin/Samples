using System;

namespace Generated
{
    public class DBErrorLogEntity
    {
        public int LogId { get; set; }
        public required string KeyName { get; set; }
        public int KeyId { get; set; }
        public int ErrorCode { get; set; }
        public int CreatedBy { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string Message { get; set; }
    }
}
