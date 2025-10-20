using System;

namespace Generated
{
    public class LoginBlackListEntity
    {
        public int LoginBlackListId { get; set; }
        public required string CountryCode { get; set; }
        public int? CurrencyId { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
    }
}
