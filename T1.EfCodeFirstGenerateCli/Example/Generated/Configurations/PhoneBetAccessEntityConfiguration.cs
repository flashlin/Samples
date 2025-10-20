using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PhoneBetAccessEntityConfiguration : IEntityTypeConfiguration<PhoneBetAccessEntity>
    {
        public void Configure(EntityTypeBuilder<PhoneBetAccessEntity> builder)
        {
            builder.ToTable("PhoneBetAccess");

            builder.HasKey(x => x.ID);

            builder.Property(x => x.ID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.key)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.account)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.expireDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.status)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.roleid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

        }
    }
}
