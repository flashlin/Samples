using System;

namespace Generated
{
    public class AccountActionLogEntity
    {
        public int ActionID { get; set; }
        public required string ActionType { get; set; }
        public required string Action { get; set; }
        public required string Actor { get; set; }
        public DateTime ModifiedDate { get; set; }
    }
}
