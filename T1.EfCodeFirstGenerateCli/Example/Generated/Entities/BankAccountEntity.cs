using System;

namespace Generated
{
    public class BankAccountEntity
    {
        public int custid { get; set; }
        public required string username { get; set; }
        public int? type { get; set; }
        public required string description { get; set; }
        public int? currency { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
