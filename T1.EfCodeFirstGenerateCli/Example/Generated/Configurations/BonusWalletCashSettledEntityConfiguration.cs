using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BonusWalletCashSettledEntityConfiguration : IEntityTypeConfiguration<BonusWalletCashSettledEntity>
    {
        public void Configure(EntityTypeBuilder<BonusWalletCashSettledEntity> builder)
        {
            builder.ToTable("BonusWalletCashSettled");

            builder.HasKey(x => x.BonusWalletId);

            builder.Property(x => x.BonusWalletId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Recommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Mrecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CashSettled)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CashReturn)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgtCashSettled)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgtCashReturn)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaCashSettled)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaCashReturn)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaCashSettled)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaCashReturn)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TransferIn)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TransferOut)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LastTransferOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
