using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class LockAccountEntityConfiguration : IEntityTypeConfiguration<LockAccountEntity>
    {
        public void Configure(EntityTypeBuilder<LockAccountEntity> builder)
        {
            builder.ToTable("LockAccount");

            builder.HasKey(x => x.username);

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.lockdate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.admin)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

        }
    }
}
