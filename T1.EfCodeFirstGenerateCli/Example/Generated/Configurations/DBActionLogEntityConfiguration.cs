using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DBActionLogEntityConfiguration : IEntityTypeConfiguration<DBActionLogEntity>
    {
        public void Configure(EntityTypeBuilder<DBActionLogEntity> builder)
        {
            builder.ToTable("DBActionLog");


            builder.Property(x => x.Action)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.ActionTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
