using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerCreditEntityConfiguration : IEntityTypeConfiguration<CustomerCreditEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerCreditEntity> builder)
        {
            builder.ToTable("CustomerCredit");

            builder.HasKey(x => x.CustId);

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AccountType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.PlayableLimit)
                .HasColumnType("decimal(19,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.PlayerMaxLimit)
                .HasColumnType("decimal(19,2)")
                .IsRequired()
            ;

            builder.Property(x => x.TableLimit)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
            ;

            builder.Property(x => x.StakeLimit)
                .HasColumnType("decimal(19,2)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LimitExpiredDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.DailyPlayerMaxLose)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.DailyPlayerMaxWin)
                .HasColumnType("decimal(5,2)")
            ;

            builder.Property(x => x.DailyResetEnabled)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.PlayableLimit1)
                .HasColumnType("decimal(19,2)")
                .HasDefaultValue(0)
            ;

        }
    }
}
