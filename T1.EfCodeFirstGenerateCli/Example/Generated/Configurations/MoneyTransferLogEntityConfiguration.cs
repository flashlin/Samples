using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MoneyTransferLogEntityConfiguration : IEntityTypeConfiguration<MoneyTransferLogEntity>
    {
        public void Configure(EntityTypeBuilder<MoneyTransferLogEntity> builder)
        {
            builder.ToTable("MoneyTransferLog");

            builder.HasKey(x => x.MTID);

            builder.Property(x => x.MTID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TransferType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.FromID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FromAccountID)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ToAccountID)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ISOCurrency)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.MarketRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.TransferStatus)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.PaymentMethod)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Description)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.IsRead)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.MTBID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TransferFollowupGroup)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

        }
    }
}
