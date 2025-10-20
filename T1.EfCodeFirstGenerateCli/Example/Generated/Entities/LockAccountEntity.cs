using System;

namespace Generated
{
    public class LockAccountEntity
    {
        public required string username { get; set; }
        public int custid { get; set; }
        public DateTime lockdate { get; set; }
        public required string admin { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
