using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class exchangeEntityConfiguration : IEntityTypeConfiguration<exchangeEntity>
    {
        public void Configure(EntityTypeBuilder<exchangeEntity> builder)
        {
            builder.ToTable("exchange");

            builder.HasKey(x => x.exchangeid);

            builder.Property(x => x.exchangeid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.currency)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.stakerate)
                .HasColumnType("")
            ;

            builder.Property(x => x.actualrate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.commaxpayout)
                .HasColumnType("")
            ;

            builder.Property(x => x.creator)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.minbet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.JoinNowMinBetDefault)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.JoinNowMaxBetDefault)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.JoinNowMaxPerMatchDefault)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

            builder.Property(x => x.ISOCode)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ISOCurrency)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.CurrencyEnabled)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.JoinNowEnabled)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.RBMinbet)
                .HasColumnType("decimal(19,2)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RToteMaxBet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.RToteMinBet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.RToteActualRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.MarketRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.AffiliateJoinNowMaxBetDefault)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.AffiliateJoinNowMaxPerMatchDefault)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.RToteMaxPerRace)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.CasinoPlayableLimit)
                .HasColumnType("decimal(19,2)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.adminFeeAmount)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.ForecastRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.RealMarketRate)
                .HasColumnType("decimal(12,8)")
            ;

        }
    }
}
