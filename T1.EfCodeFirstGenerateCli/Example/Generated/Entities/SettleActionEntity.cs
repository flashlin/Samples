using System;

namespace Generated
{
    public class SettleActionEntity
    {
        public int ActionID { get; set; }
        public required string StoredProcedure { get; set; }
        public required string ParamNames { get; set; }
        public required string ParamValues { get; set; }
        public int Creator { get; set; }
        public DateTime CreatedOn { get; set; }
        public DateTime TStamp { get; set; }
    }
}
