using System;

namespace Generated
{
    public class exchangeEntity
    {
        public int exchangeid { get; set; }
        public required string currency { get; set; }
        public required string stakerate { get; set; }
        public decimal? actualrate { get; set; }
        public required string commaxpayout { get; set; }
        public required string creator { get; set; }
        public decimal? minbet { get; set; }
        public decimal? JoinNowMinBetDefault { get; set; }
        public decimal? JoinNowMaxBetDefault { get; set; }
        public decimal? JoinNowMaxPerMatchDefault { get; set; }
        public DateTime tstamp { get; set; }
        public int? ISOCode { get; set; }
        public required string ISOCurrency { get; set; }
        public bool? CurrencyEnabled { get; set; }
        public bool? JoinNowEnabled { get; set; }
        public decimal? RBMinbet { get; set; }
        public decimal? RToteMaxBet { get; set; }
        public decimal? RToteMinBet { get; set; }
        public decimal? RToteActualRate { get; set; }
        public decimal? MarketRate { get; set; }
        public decimal? AffiliateJoinNowMaxBetDefault { get; set; }
        public decimal? AffiliateJoinNowMaxPerMatchDefault { get; set; }
        public decimal? RToteMaxPerRace { get; set; }
        public decimal? CasinoPlayableLimit { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public decimal? adminFeeAmount { get; set; }
        public decimal? ForecastRate { get; set; }
        public decimal? RealMarketRate { get; set; }
    }
}
