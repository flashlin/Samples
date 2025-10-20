using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MpCashSettledEntityConfiguration : IEntityTypeConfiguration<MpCashSettledEntity>
    {
        public void Configure(EntityTypeBuilder<MpCashSettledEntity> builder)
        {
            builder.ToTable("MpCashSettled");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CashSettled)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.CashReturn)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.AgtCashSettled)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.AgtCashReturn)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.MaCashSettled)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.MaCashReturn)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SmaCashSettled)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SmaCashReturn)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.TransferIn)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.TransferOut)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.ServiceProvider)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.LastTransferOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CurrencyId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Username)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

        }
    }
}
