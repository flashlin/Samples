using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CurrencyTransactionsEntityConfiguration : IEntityTypeConfiguration<CurrencyTransactionsEntity>
    {
        public void Configure(EntityTypeBuilder<CurrencyTransactionsEntity> builder)
        {
            builder.ToTable("CurrencyTransactions");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TransDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.FromAccount)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToAccount)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ExchangeRate)
                .HasColumnType("decimal(16,8)")
                .IsRequired()
            ;

            builder.Property(x => x.TransDesc)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.TransRemark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.ExchangeDescription)
                .HasColumnType("nvarchar(300)")
                .IsRequired()
                .HasMaxLength(300)
            ;

            builder.Property(x => x.ExchangeRemark)
                .HasColumnType("nvarchar(255)")
                .IsRequired()
                .HasMaxLength(255)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.StatementType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.FromCurrency)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(500)")
                .HasMaxLength(500)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToCurrency)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Jurisdiction)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

        }
    }
}
