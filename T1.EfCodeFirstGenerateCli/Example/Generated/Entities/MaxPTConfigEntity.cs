using System;

namespace Generated
{
    public class MaxPTConfigEntity
    {
        public int Id { get; set; }
        public int CurrencyId { get; set; }
        public required string Currency { get; set; }
        public byte ProductType { get; set; }
        public byte AccountType { get; set; }
        public decimal MaxPT { get; set; }
        public DateTime ModifiedDate { get; set; }
        public int? LeoEnumValue { get; set; }
        public int? SmaId { get; set; }
    }
}
