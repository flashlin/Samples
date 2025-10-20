using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class GamesCashUsedTransactionEntityConfiguration : IEntityTypeConfiguration<GamesCashUsedTransactionEntity>
    {
        public void Configure(EntityTypeBuilder<GamesCashUsedTransactionEntity> builder)
        {
            builder.ToTable("GamesCashUsedTransaction");

            builder.HasKey(x => new { x.TransactionId, x.ExternalRefNo });

            builder.Property(x => x.TransactionId)
                .HasColumnType("varchar(110)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(110)
            ;

            builder.Property(x => x.ExternalRefNo)
                .HasColumnType("varchar(110)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(110)
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.IsVerified)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.AgtPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(-1)
            ;

            builder.Property(x => x.MaPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(-1)
            ;

            builder.Property(x => x.SmaPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(-1)
            ;

            builder.Property(x => x.BettingBudgetAmount)
                .HasColumnType("decimal(19,6)")
            ;

        }
    }
}
