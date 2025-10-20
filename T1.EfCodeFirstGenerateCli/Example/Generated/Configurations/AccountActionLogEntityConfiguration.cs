using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AccountActionLogEntityConfiguration : IEntityTypeConfiguration<AccountActionLogEntity>
    {
        public void Configure(EntityTypeBuilder<AccountActionLogEntity> builder)
        {
            builder.ToTable("AccountActionLog");


            builder.Property(x => x.ActionID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ActionType)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Action)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Actor)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

        }
    }
}
