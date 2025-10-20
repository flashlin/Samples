using System;

namespace Generated
{
    public class ExternalExchangeEntity
    {
        public DateTime EffectiveDate { get; set; }
        public required string SystemRate { get; set; }
        public required string ExternalRate { get; set; }
        public int Currency { get; set; }
        public required string CurrencyStr { get; set; }
        public DateTime ModifiedDate { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
