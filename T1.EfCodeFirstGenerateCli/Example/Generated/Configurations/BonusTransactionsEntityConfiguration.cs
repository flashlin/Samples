using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BonusTransactionsEntityConfiguration : IEntityTypeConfiguration<BonusTransactionsEntity>
    {
        public void Configure(EntityTypeBuilder<BonusTransactionsEntity> builder)
        {
            builder.ToTable("BonusTransactions");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.WinLostDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.FromAccountId)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToAccountId)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.FromCustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ToCustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.FromCurrency)
                .HasColumnType("varchar(10)")
                .IsRequired()
                .HasMaxLength(10)
            ;

            builder.Property(x => x.ToCurrency)
                .HasColumnType("varchar(10)")
                .IsRequired()
                .HasMaxLength(10)
            ;

            builder.Property(x => x.FromActualRate)
                .HasColumnType("decimal(12,8)")
                .IsRequired()
            ;

            builder.Property(x => x.ToActualRate)
                .HasColumnType("decimal(12,8)")
                .IsRequired()
            ;

            builder.Property(x => x.ServiceProvider)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.Description)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.IsSendPersonalMessage)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.DescriptionTranslated)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.RequestIdentifier)
                .HasColumnType("varchar(230)")
                .HasMaxLength(230)
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(3)
            ;

        }
    }
}
