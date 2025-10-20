using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class TracDelayEntityConfiguration : IEntityTypeConfiguration<TracDelayEntity>
    {
        public void Configure(EntityTypeBuilder<TracDelayEntity> builder)
        {
            builder.ToTable("TracDelay");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.FromAccountId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.FromCustomerID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreateDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.WinlostDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ToAccountId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToCustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ExchangeRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.Description)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.UpdatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.UpdatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.FromBankFee)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ToBankFee)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.IsDeductedFromSender)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.IsAutoInsertCurrencyTrac)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.CurrencyTracDescription)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

        }
    }
}
